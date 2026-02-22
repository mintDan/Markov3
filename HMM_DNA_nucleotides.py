import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

 
os.system("cls")


print("============================== HMM ==============================")

if __name__ == "__main__":
    #Her, lav example med High/Low GC content vs GCAT nucleotides
    #Det er 2 states, med 4 possible emissions?
    #https://www.cis.upenn.edu/~cis2620/notes/Example-Viterbi-DNA.pdf

    p = np.array([[0.5,0.5],
                  [0.4,0.6]
                  ])

    #emission to GCAT
    e = np.array([[0.2,0.3,0.3,0.2],
                  [0.3,0.2,0.2,0.3],
                  ])

     

    #p = p.T

    #e = e.T

     
    #HL 
    #ACGT

     

    #GGCACTGAA

     

    O = np.array([2,2,1,0,1,3,2,0,0])
    #O = np.array([0,0,0,0,0])
    #O = np.array([1,1,1,1,1])
    #O = np.array([2,2,2,2,2])
    #O = np.array([3,3,3,3,3])
    #O = np.array([2,2,2,2,2,2,2])

    #Vi observe RBR,

    #R = 0

    #B = 1

    #O = np.array([0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0])

    #O = np.array([1,1,1,1,1,1])

     

    #O = np.array([0,0,0,0,0])

    #Greatest chance to observe a red ball, comes from state 2.

    #Likewise, if you are in state 2, you are most likely to remain in state 2. So output of [2,2,2...] makes sense.

     

     

    #

     

    S = 2

    T = len(O)

     

     

    prob = np.zeros((T,S))

    prev = np.zeros((T,S))

     

    #initialize

    init = np.array([0.5,0.5])

    #

    for s in range(S):

        prob[0,s] = init[s]*e[s,O[0]]

     
    print("prob")
    print(prob)

     

    #t = 1

     

    for t in range(1,T):

        for s in range(S):

            for r in range(S):

                new_prob = prob[t-1,r]*p[r,s]*e[s,O[t]]

                if new_prob > prob[t,s]:

                    prob[t,s] = new_prob

                    prev[t,s] = r

     
    print("prob")
    print(prob)
    print("prev")
    print(prev)

     

    print("max prob")

    print(np.max(prob[T-1,:]))

    #np.argmax(prob[T-1,:])

     

     

    path = np.zeros(T)

    path[T-1] = np.argmax(prob[T-1,:])

    #print(path)

    for t in np.array([i for i in range(T-1)])[::-1]:

        #print(t)

        path[t] = prev[t+1,int(path[t+1])]
    print("path")
    print(path)

     

     

     

     

     

     

     

    print("==== simul ====")

    #Vi laver lige transposed, det virker til at give bedre resultat

    #Lav noget simul også...
    Nt = 1000000
    st = np.zeros(Nt)
    ot = np.zeros(Nt)

    for nt in range(Nt-1):

        RNG = np.random.uniform(0,1)
        #If we are in state 0 then
        if st[nt] == 0:

            if RNG <= p[0,0]:
                st[nt+1] = 0
            else:
                st[nt+1] = 1

        elif st[nt] == 1:
            if RNG <= p[0,1]:
                st[nt+1] = 0
            else:
                st[nt+1] = 1
            
       # pass

    print(st)

    #Her simulate emission observed states

    for nt in range(Nt):
        RNG = np.random.uniform()
        if st[nt] == 0:
            if RNG <= e[0,0]:
                ot[nt] = 0 #red
            elif RNG <= e[0,1]+e[0,0]:
                ot[nt] = 1
            elif RNG <= e[0,2]+e[0,1]+e[0,0]:
                ot[nt] = 2
            else:
                ot[nt] = 3 
        elif st[nt] == 1:
            if RNG <= e[1,0]:
                ot[nt] = 0
            elif RNG <= e[1,1]+e[1,0]:
                ot[nt] = 1
            elif RNG <= e[1,2]+e[1,1]+e[1,0]:
                ot[nt] = 2
            else:
                ot[nt] = 3

    print(ot)

        
    fig,ax = plt.subplots()
    plt.title("Hidden states [0,1,2]")
    plt.plot(st[:100])

    plt.xlabel("t")
    plt.ylabel("state")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0,2])
    plt.savefig("3statesimulation.png")
    plt.show()


    ind0= ot[:100] == 0
    ind1 = ot[:100] == 1
    ind2 = ot[:100] == 2
    ind3 = ot[:100] == 3
    #o_b = 
    #o_r = ot == 0
    indx = np.array([i for i in range(100)])
    fig,ax = plt.subplots()
    plt.title("Observable states [0,1]")


    plt.scatter(indx[ind0],ot[:100][ind0],color="red")
    plt.scatter(indx[ind1],ot[:100][ind1],color="blue")
    plt.scatter(indx[ind2],ot[:100][ind2],color="blue")
    plt.scatter(indx[ind3],ot[:100][ind3],color="blue")
    plt.xlabel("t")
    plt.ylabel("state")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0,4])
    plt.savefig("2observablestatessimulation.png")
    plt.show()

    #p = p.T


    print(p)
    eigval,eigvec = np.linalg.eig(p)

    print("eigval")

    print(eigval)

    print("eigvec")
    print(eigvec)

    count0 = np.sum(st == 0)
    count1 = np.sum(st == 1)


    print("countvec")
    #print(st == 2)
    print(count0/Nt,count1/Nt)
    countvec = np.array([count0/Nt,count1/Nt])
    print(countvec)
    print(countvec/np.linalg.norm(countvec))
     

     

    a=[1,2]

    print(id(a[0]))

    #4318513456

    print(id(a[1]))

    #4318513488

    a=[1,'a',3]

    print(id(a[0]))

    #4318513456

    print(id(a[1]))

    #4319642992

    print(id(a[2]))

    #4318513520



    import networkx as nx


    fig,ax = plt.subplots(figsize=(20,20))


    DG = nx.DiGraph()
    DG.add_node(1,pos=(1,1),color="white")
    DG.add_node(2,pos=(10,1),color="white")
    DG.add_edge(2, 1,weight=p[1,0],label_pos=0.5)   # adds the nodes in order 2, 1
    DG.add_edge(1, 2,weight=p[0,1],label_pos=0.5)
    DG.add_edge(1, 1,weight=p[0,0],label_pos=0.5)
    DG.add_edge(2, 2,weight=p[1,1],label_pos=0.5)

    #pos = nx.spring_layout(DG)
    pos=nx.get_node_attributes(DG,'pos')
    options = {
        "font_size": 44,
        "node_size": 5000,
        "node_color": "white",
        "edgecolors": "black",#,
        "linewidths": 8,
        "width": 8,
        "alpha": 0.8
    }

    #bboxlbl = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0),alpha=1)
    bboxlbl = dict(boxstyle="round", ec="white", fc="white",alpha=0.6)
    labels = nx.get_edge_attributes(DG,'weight')

    #https://stackoverflow.com/questions/74350464/how-to-better-visualize-networkx-self-loop-plot

    nx.draw(DG, with_labels=True, pos=pos,font_weight='bold',ax=ax,connectionstyle='arc3, rad = 0.1',**options)#,arrows=True)
    #e = nx.draw_networkx_edges(DG, pos = pos)#, edge_color='g', arrowsize=50, node_size = 800)
    nx.draw_networkx_edge_labels(DG,pos=pos, edge_labels=labels,connectionstyle='arc3, rad = 0.1',ax=ax,
                                 #label_pos=0.3,
                                 verticalalignment="top",
                                 horizontalalignment="left",bbox = bboxlbl,font_size=40)

    plt.tight_layout()
    plt.savefig("pgraph.png")
    plt.show()


    fig,ax = plt.subplots(figsize=(20,20))



    DG = nx.DiGraph()
    DG.add_node(1,pos=(1,2),color="white")
    DG.add_node(2,pos=(1,3),color="white")

    DG.add_node("A",color="lightblue",pos=(2,1.5))
    DG.add_node("C",color="lightblue",pos=(2,2.5))
    DG.add_node("G",color="lightblue",pos=(2,3.5))
    DG.add_node("T",color="lightblue",pos=(2,4.5))
    DG.add_edge(1, "A",weight=round(e[0,0],2),label_pos=0.5)   # adds the nodes in order 2, 1
    DG.add_edge(1, "C",weight=round(e[0,1],2),label_pos=0.5)
    DG.add_edge(1, "G",weight=round(e[0,2],2),label_pos=0.5)   # adds the nodes in order 2, 1
    DG.add_edge(1, "T",weight=round(e[0,3],2),label_pos=0.5)
    DG.add_edge(2, "A",weight=round(e[1,0],2),label_pos=0.5)   # adds the nodes in order 2, 1
    DG.add_edge(2, "C",weight=round(e[1,1],2),label_pos=0.5)
    DG.add_edge(2, "G",weight=round(e[1,2],2),label_pos=0.5)   # adds the nodes in order 2, 1
    DG.add_edge(2, "T",weight=round(e[1,3],2),label_pos=0.5)


    color_map = ["white","white","lightblue","lightblue","lightblue","lightblue"]
    #color_map = None
    #pos = nx.spring_layout(DG)
    pos=nx.get_node_attributes(DG,'pos')
    options = {
        "font_size": 44,
        "node_size": 5000,
        "node_color": "white",
        "edgecolors": "black",#,
        "linewidths": 8,
        "width": 8,
        "alpha": 0.8
    }


    bboxlbl = dict(boxstyle="round", ec="white", fc="white",alpha=0.6)
    labels = nx.get_edge_attributes(DG,'weight')

    #https://stackoverflow.com/questions/74350464/how-to-better-visualize-networkx-self-loop-plot

    nx.draw(DG, with_labels=True, pos=pos,font_weight='bold',ax=ax,connectionstyle='arc3, rad = 0.1',**options)#,arrows=True)

    nx.draw_networkx_nodes(DG, pos,node_color=color_map
                            ,node_size=5000)#, nodelist=[0, 1, 2, 3], node_color="tab:red", **options)

    nx.draw_networkx_edge_labels(DG,pos=pos, edge_labels=labels,connectionstyle='arc3, rad = 0.1',ax=ax,
                                 label_pos=0.3,
                                 verticalalignment="top",
                                 horizontalalignment="left",bbox = bboxlbl,font_size=40)

    plt.tight_layout()
    plt.axis("off")
    plt.savefig("egraph.png")
    plt.show()
     
     
     
     
    print("plot path")
    fig,ax = plt.subplots()
    print("path")
    print(path)
    print("O")
    print(O)
    print("max prob")
    print(np.max(prob[T-1,:]))


    Npath = len(path)
    bbox1=dict(facecolor='white', edgecolor='black', boxstyle="circle,pad=0.3",lw=1,alpha=0.5)
    bbox2=dict(facecolor='white', edgecolor='black', boxstyle="circle,pad=0.3",lw=4)
    bboxb=dict(facecolor='lightblue', edgecolor='black', boxstyle="circle,pad=0.3",lw=1)
    bboxr=dict(facecolor='lightblue', edgecolor='black', boxstyle="circle,pad=0.3",lw=1)

    yranges = [0, 0.5]
    newpath = np.zeros(Npath)
    for k in range(Npath):
        if path[k] == 0:
            newpath[k] = yranges[0]
        elif path[k] == 1:
            newpath[k] = yranges[1]
        #else:
       #     newpath[k] = yranges[2]

    print("newpath")
    print(newpath)
    plt.plot(newpath,color="black")
    #ACGT
    fontsizenode = 18
    for k in range(Npath):
        if O[k] == 0:
            ax.text(k,2,"A",bbox = bboxb,va="center",ha="center",fontsize=fontsizenode)
        elif O[k] == 1:
            ax.text(k,2,"C",bbox = bboxb,va="center",ha="center",fontsize=fontsizenode)
        elif O[k] == 2:
            ax.text(k,2,"G",bbox = bboxb,va="center",ha="center",fontsize=fontsizenode)
        else:
            ax.text(k,2,"T",bbox = bboxb,va="center",ha="center",fontsize=fontsizenode)

    dy = 0.5


    Slist = ["H","L"]
    for k in range(Npath):
        for i in range(S):
            print(path[k],yranges[i])
            if i == path[k]:
                ax.text(k,yranges[i],Slist[i],bbox = bbox2,va="center",ha="center",fontsize=fontsizenode)
            else:
                ax.text(k,yranges[i],Slist[i],bbox = bbox1,va="center",ha="center",fontsize=fontsizenode,alpha=0.5)
                #ax.text(k,1,"U3",bbox = bbox1)
                
                
    plt.axis([-1,Npath+1,0,3])
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("path"+"".join([str(int(i)) for i in O])+".png")
    plt.show()
     
