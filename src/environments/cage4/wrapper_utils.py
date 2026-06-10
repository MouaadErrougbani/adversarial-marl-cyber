import numpy as np
import torch

from src.environments.cage4.constants import (
    SN_BLOCK_SIZE,
    INTERNET,
)

from src.environments.cage4.topology import (
    ROUTERS,
    ACCESSABLE_OFFLINE,
)


def parse_tabular(x, g):
        '''
        Pull the per-host data out of the tabular observation. 
        Data is formatted as 
        
        Mission phase           (1d, 0,1, or 2)
        Subnet info (x8)
            Subnet id           (9d)
            Blocked subnets     (9d)
            Comm policy         (9d)
            Comprimised hosts   (16d)
            Scanned hosts       (16d)
        Messages (4x8)

        We pull out the host data and append it to the host nodes, 
        pass the phase as a global vector, 
        and add messages to the subnet nodes they originate from 
        '''
        # First bit is phase
        # Last 4x8 are messages from other agents
        phase_idx = int(x[0])
        sn_block = x[1:-(4*8)]
        subnets = sn_block.shape[0] // SN_BLOCK_SIZE

        relevant_subnets = []
        src = []
        dst = []
        x = torch.zeros(g.n_permenant_nodes, 2)
        msgs = []

        # Only affects agent4 but may as well be generalizeable
        for i in range(subnets):
            block = sn_block[SN_BLOCK_SIZE*i : SN_BLOCK_SIZE*(i+1)]

            # Pull out edges between subnets
            sn = block[:18]
            me = ROUTERS[ sn[:9].nonzero()[0][0] ]
            can_maybe_connect_to = (sn[9:18] == 0).nonzero()[0]

            # Logic for subnet routing 
            if INTERNET in can_maybe_connect_to:
                can_connect_to = [ROUTERS[i] for i in can_maybe_connect_to]
            else:
                # Can connect to anything in LAN
                can_connect_to = [
                    ROUTERS[i] for i in can_maybe_connect_to
                    if ROUTERS[i] in ACCESSABLE_OFFLINE[me]
                ]

            router_name = me
            me = [me] * len(can_connect_to)
            src += can_connect_to
            dst += me

            # Pull out features for servers/hosts that exist
            hosts = torch.from_numpy(block[27:]).reshape(2,16).T
            n_srv, n_usr = g.subnet_size[router_name]
            srv_idx = list(range(n_srv))
            usr_idx = list(range(6,n_usr+6))

            # Insert into rows corresponding w server/host nodes in graph
            # (Always directly after node for subnet they are on)
            start_usr_idx = g.nids[router_name]+1
            start_srv_idx = start_usr_idx + len(usr_idx)
            end_srv_idx = start_srv_idx + len(srv_idx)

            # Note: TabularWrapper goes from server to host, but
            # graph goes from host to server (alphabetically)
            # so we have to do some lifting to rearrange
            x[start_usr_idx : start_srv_idx] = hosts[usr_idx]
            x[start_srv_idx : end_srv_idx] = hosts[srv_idx]

            # Each subnet can add 2 bits to the message for if any hosts
            # are compromised/have been scanned
            msg = list((hosts.sum(dim=0) > 0).long())
            msgs += msg

            relevant_subnets.append(router_name)

        g.set_firewall_rules(src,dst)
        phase = torch.zeros((1,3))
        phase[0,phase_idx] = 1

        # Make messages all 8-dim and add checkbit to the end
        padding = 8-len(msgs)
        msgs += [0]*padding
        msgs[-1] = 1
        msg = np.array(msgs)

        return x,phase,msg

def combine_data(graph_x, tabular_x):
        '''
        Stick the tabular data onto the node feature matrix 
        on the appropriate rows--those corresponding with 
        the hosts the tabular data is referencing 
        '''
        # Tabular x only accounts for subnets and workstations
        # Processes/connections have higher indices, but no features
        # from the FlatActionWrapper, so need to be padded before combined
        padding = torch.zeros(
            graph_x.size(0) - tabular_x.size(0),
            tabular_x.size(1)
        )
        tabular_x = torch.cat([tabular_x, padding], dim=0)

        return torch.cat([graph_x, tabular_x], dim=1)

def to_obs(x,ei,masks,phase, other_msg,my_msg):
        '''
        Prepare for GNN injestion (assumes unbatched. E.g. this is
        called during inference, not training)

        Args:  
            x: feature matrix           (Nxd tensor)
            ei: edge index              (2xE tensor)
            masks: list of bitmaps for servers, users, 
                   subnet edges, and routers 
            phase: global state vector  (1x3 tensor)
            other_msg: messages from other agents
                                        (4x8 tensor)
            my_msg: the message this agent sends to 
                    other agents        (1x8 tensor)
        '''

        # Happens in all cases except agent_4
        if len(masks) == 1:
            (srv,usr,edge,rtrs) = masks[0]

            all_msg = torch.zeros((x.size(0), 3))
            all_msg[rtrs] = torch.from_numpy(other_msg).float()

            # Edge[0][0] is always the subnet node managed by this agent
            all_msg[edge[0][0], :2] = torch.from_numpy(my_msg[:2]).float()

            # Set 'is_recieved' to a special value to indicate this is self
            all_msg[edge[0][0], 2] = -1

            x = torch.cat([x, all_msg], dim=1)

            return (
                x,ei,phase,
                srv,torch.tensor([srv.size(0)]),
                usr,torch.tensor([usr.size(0)]),
                edge, False
            )

        # If this is agent 4's state, we have to be careful to batch 
        # three observations together. The graph info is the same, 
        # so we don't duplicate that, but we have to concat the 
        # masks together and scatter the messages we recieved
        # properly to the feature matrix
        srv,usr,edges = [],[],[]
        n_srv,n_usr = [],[]
        my_ids = []
        for (s,u,e,_) in masks:
            my_ids.append(e[0][0].item())

            srv.append(s)
            usr.append(u)
            edges.append(e)

            n_srv.append(s.size(0))
            n_usr.append(u.size(0))

        rtrs = masks[0][3]
        other_rtrs = [o.item() for o in rtrs if o.item() not in my_ids]
        all_msg = torch.zeros(x.size(0), 3)

        all_msg[other_rtrs] = torch.from_numpy(other_msg).float()
        all_msg[my_ids, :2] = torch.from_numpy(my_msg[:6].reshape(3,2)).float()
        all_msg[my_ids, 2] = -1

        x = torch.cat([x, all_msg], dim=1)

        return (
            x,ei,phase.repeat_interleave(3,0),
            torch.cat(srv), torch.tensor(n_srv),
            torch.cat(usr), torch.tensor(n_usr),
            torch.cat(edges, dim=1), True
        )
