# Python program for clustering by modifying Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

def return_key(diction, val):
    ret_list = []
    for i in range(len(diction)):
        if list(diction.values())[i]==val:
            ret_list.append(list(diction.keys())[i])
    return ret_list

# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = [] # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def KruskalMSTClusters(self, K):
        ind = 0 # An index variable, used as a pointer for sorted edges
        e = 0 # Last cluster
        self.graph = sorted(self.graph,key=lambda item: item[2])
        self.cluster_dict = {}
        for i in range(K):
            self.cluster_dict[i] = []
        self.mach_cluster_dict = {}
        for i in range(self.V):
            self.mach_cluster_dict[str(i)]=None
        if self.V >= K > 1:
            N = self.V
            while K != N:
                u,v,w = self.graph[ind]
                valuesList = list(self.cluster_dict.values())
                flat_list = [item for sublist in valuesList for item in sublist]
                if u not in flat_list and v not in flat_list: # None of the machines are in cluster_dict - assign to a new cluster
                    if e < K:
                        print(u, " and ", v, " are not assigned to clusters")
                        self.cluster_dict[e].append(u)
                        self.cluster_dict[e].append(v)
                        self.mach_cluster_dict[str(u)] = e
                        self.mach_cluster_dict[str(v)] = e
                        e += 1 # Indicates that items are added to a new cluster
                        N -= 1 # Indicates that the no.of currently formed clusters decreased by 1
                        ind +=1 # Indicates to point to the next shortest edge
                    else:
                        print("Can not create a new cluster. The required no.of clusters already formed.")
                        ind += 1  # Indicates to point to the next shortest edge
                elif u in flat_list and v not in flat_list: # One of the machines is in cluster_dict - assign to that cluster
                    print(v, " not assigned to a cluster")
                    for k in range(K):
                        if u in self.cluster_dict[k]:
                            self.cluster_dict[k].append(v)
                            self.mach_cluster_dict[str(v)] = k
                            N -= 1 # Indicates that the no.of currently formed clusters decreased by 1
                            ind += 1 # Indicates to point to the next shortest edge
                            break
                elif v in flat_list and u not in flat_list: # One of the machines is in cluster_dict - assign to that cluster
                    print(u, " not assigned to a cluster")
                    for k in range(K):
                        if v in self.cluster_dict[k]:
                            self.cluster_dict[k].append(u)
                            self.mach_cluster_dict[str(u)] = k
                            N -= 1 # Indicates that the no.of currently formed clusters decreased by 1
                            ind += 1 # Indicates to point to the next shortest edge
                            break
                else:
                    print("cycle detected")
                    ind += 1 # Indicates to point to the next shortest edge

            # Clusters have formed, but some machines are not assigned to a cluster
            # Check if the no.of clusters without machines assigned is greater than or equal to the no.of machines without a cluster assigned. Then, assign 1 machine per cluster
            empty_clusters_list = return_key(self.cluster_dict, [])
            clusterless_machs_list = return_key(self.mach_cluster_dict, None)
            if len(empty_clusters_list) >= len(clusterless_machs_list):
                for i in clusterless_machs_list:
                    self.mach_cluster_dict[i]=e
                    self.cluster_dict[e].append(int(i))
                    e += 1

        return (self.cluster_dict, self.mach_cluster_dict)

if __name__ == '__main__':
    # Test case 1
    # g = Graph(4)
    # g.addEdge(0, 1, 10)
    # g.addEdge(0, 2, 6)
    # g.addEdge(0, 3, 5)
    # g.addEdge(1, 3, 15)
    # g.addEdge(2, 3, 4)

    # Test case 2
    # g = Graph(6)
    # g.addEdge(0, 1, 4)
    # g.addEdge(0, 2, 4)
    # g.addEdge(1, 2, 2)
    # g.addEdge(2, 3, 3)
    # g.addEdge(2, 5, 2)
    # g.addEdge(2, 4, 4)
    # g.addEdge(3, 4, 3)
    # g.addEdge(5, 4, 3)

    # Test case 3
    g = Graph(7)
    g.addEdge(0, 1, 4)
    g.addEdge(0, 2, 4)
    g.addEdge(1, 2, 2)
    g.addEdge(2, 3, 3)
    g.addEdge(2, 5, 2)
    g.addEdge(2, 4, 4)
    g.addEdge(3, 4, 3)
    g.addEdge(5, 4, 3)
    g.addEdge(6, 4, 13)

    # Function call
    g.KruskalMSTClusters(K=4)