class NodeTracker:
    '''
    Just a hash map with extra steps
    '''
    def __init__(self):
        self.nid = 0
        self.mapping = dict()
        self.inv_mapping = dict()

    def __getitem__(self, node_str):
        node_str = str(node_str)
        if (nid := self.mapping.get(node_str)) is not None:
            return nid

        # Add to dict if it doesn't exist
        self.mapping[node_str] = self.nid
        self.inv_mapping[self.nid] = node_str

        self.nid += 1
        return self.mapping[node_str]

    def pop(self, node_str):
        nid = self.mapping.get(node_str, None)
        if nid:
            self.mapping.pop(node_str)
            self.inv_mapping.pop(nid)

    def get(self, node_str):
        return self.mapping.get(node_str, None)

    def id_to_str(self, nid):
        return self.inv_mapping.get(nid)

    def names(self):
        return list(self.mapping.keys())
