from pprint import pprint

from CybORG.Simulator.Actions.AbstractActions import (
    Remove,
    Restore,
)

from CybORG.Simulator.Actions.ConcreteActions.DecoyActions import (
    DeployDecoy,
)

from CybORG.Shared.Enums import TernaryEnum

from src.observations.graph.nodes import (
    ConnectionNode,
    FileNode,
    init_decoy,
)


class GraphUpdatesMixin:
    """
    Gestion des mises à jour dynamiques du graphe.
    """

    def parse_observation(self, obs):

        success = obs.pop("success")

        if "action" in obs:
            act = obs.pop("action")
        else:
            act = None

        if isinstance(act, Restore) and success == TernaryEnum.TRUE:

            host_id = self.nids[act.hostname]

            host_ports = [
                k
                for (k, v) in self.nids.inv_mapping.items()
                if v.startswith(act.hostname)
                and k != host_id
            ]

            removed = [host_id] + host_ports

            new_edges = [[], []]

            for i in range(
                len(self.transient_edges[0])
            ):
                src = self.transient_edges[0][i]
                dst = self.transient_edges[1][i]

                if (
                    src not in removed
                    and dst not in removed
                ):
                    new_edges[0].append(src)
                    new_edges[1].append(dst)

            self.transient_edges = new_edges

            if act.hostname in self.host_to_sussy:
                self.host_to_sussy.pop(
                    act.hostname
                )

        elif (
            isinstance(act, Remove)
            and success == TernaryEnum.TRUE
        ):

            if act.hostname in self.host_to_sussy:
                sus_ids = self.host_to_sussy.pop(
                    act.hostname
                )
            else:
                sus_ids = []

            if sus_ids:

                new_edges = [[], []]

                for i in range(
                    len(self.transient_edges[0])
                ):
                    src = self.transient_edges[0][i]
                    dst = self.transient_edges[1][i]

                    if (
                        src not in sus_ids
                        and dst not in sus_ids
                    ):
                        new_edges[0].append(src)
                        new_edges[1].append(dst)

                self.transient_edges = new_edges

        elif (
            isinstance(act, DeployDecoy)
            and success == TernaryEnum.TRUE
        ):

            proc = obs[act.hostname][
                "Processes"
            ][0]

            service = proc["service_name"]

            port_num = self.DECOY_TO_PORT[
                service
            ]

            host_id = self.nids[
                act.hostname
            ]

            port_id = self.nids[
                f"{act.hostname}:{port_num}"
            ]

            self.nodes[port_id] = (
                init_decoy(
                    port_id,
                    service,
                )
            )

            self.transient_edges[0].append(
                port_id
            )

            self.transient_edges[1].append(
                host_id
            )

        edges = set()

        for hostname, info in obs.items():

            host_id = self.nids[hostname]

            if (
                procs := info.get(
                    "Processes"
                )
            ):
                for proc in procs:

                    conn = proc.get(
                        "Connections"
                    )

                    if conn is None:
                        continue

                    if len(conn) > 1:
                        pprint(proc)
                        raise ValueError()

                    conn = conn[0]

                    local_addr = conn.get(
                        "local_address"
                    )

                    local_port = conn.get(
                        "local_port"
                    )

                    remote_addr = conn.get(
                        "remote_address"
                    )

                    remote_port = conn.get(
                        "remote_port"
                    )

                    if not (
                        local_addr
                        and remote_addr
                    ):
                        continue

                    local_host = self.ip_map[
                        local_addr
                    ]

                    remote_host = self.ip_map[
                        remote_addr
                    ]

                    lh_id = self.nids[
                        local_host
                    ]

                    rh_id = self.nids[
                        remote_host
                    ]

                    if local_port:

                        lp_name = (
                            f"{local_host}:"
                            f"{local_port}"
                        )

                        lp_id = self.nids[
                            lp_name
                        ]

                        if (
                            local_port > 49152
                            or self.nodes.get(
                                lp_id
                            )
                            is None
                        ):
                            self.nodes[
                                lp_id
                            ] = ConnectionNode(
                                lp_id,
                                is_ephemeral=
                                local_port
                                > 49152,
                            )

                        self.transient_edges[
                            0
                        ] += [
                            rh_id,
                            lp_id,
                        ]

                        self.transient_edges[
                            1
                        ] += [
                            lp_id,
                            lh_id,
                        ]

                    if remote_port:

                        rp_name = (
                            f"{remote_host}:"
                            f"{remote_port}"
                        )

                        rp_id = self.nids[
                            rp_name
                        ]

                        if (
                            remote_port > 49152
                            or self.nodes.get(
                                rp_id
                            )
                            is None
                        ):
                            self.nodes[
                                rp_id
                            ] = ConnectionNode(
                                rp_id,
                                is_ephemeral=
                                remote_port
                                > 49152,
                            )

                        self.transient_edges[
                            0
                        ] += [
                            lh_id,
                            rp_id,
                        ]

                        self.transient_edges[
                            1
                        ] += [
                            rp_id,
                            rh_id,
                        ]

            if (
                files := info.get(
                    "Files"
                )
            ):
                for file in files:

                    file_uq_str = (
                        f"{hostname}:"
                        f"{file['Path']}\\"
                        f"{file['File Name']}"
                    )

                    file_id = self.nids[
                        file_uq_str
                    ]

                    self.nodes[file_id] = (
                        FileNode(
                            file_id,
                            file,
                        )
                    )

                    edges.update(
                        [
                            (
                                host_id,
                                file_id,
                            ),
                            (
                                file_id,
                                host_id,
                            ),
                        ]
                    )

        if edges:

            src, dst = zip(*edges)

            self.transient_edges[0] += src
            self.transient_edges[1] += dst

        obs["success"] = success