import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

if __name__ == "__main__":
    


    os.system("cls")

    print("============================== HMM ==============================")

    #lav den med 3 urns og 2 balls Red vs Blue

    #https://www.cs.cornell.edu/~ginsparg/physics/INFO295/vit.pdf

     

     

     

    p = np.array([[0.3,0.6,0.1],

                  [0.5,0.2,0.3],

                  [0.4,0.1,0.5]])

     

    #emission to Red, Blue

    e = np.array([[0.5,0.5],

                  [1/3,2/3],

                  [3/4,1/4]])

     

     

     

     

    #Vi observe RBR,

    #R = 0

    #B = 1

    #O = np.array([0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0])

    #O = np.array([1,1,1,1,1,1])

    O = np.array([1,0,1,0,1,0]) 

    #O = np.array([0,0,0,0,0,0])

    #Greatest chance to observe a red ball, comes from state 2.

    #Likewise, if you are in state 2, you are most likely to remain in state 2. So output of [2,2,2...] makes sense.

     

     

    #

     

    S = 3

    T = len(O)

     

     

    prob = np.zeros((T,S))

    prev = np.zeros((T,S))

     

    #initialize

    init = np.array([1/3,1/3,1/3])

     

    for s in range(S):

        prob[0,s] = init[s]*e[s,O[0]]

     

    print(prob)

     

    #t = 1

     

    for t in range(1,T):

        for s in range(S):

            for r in range(S):

                new_prob = prob[t-1,r]*p[r,s]*e[s,O[t]]

                if new_prob > prob[t,s]:

                    prob[t,s] = new_prob

                    prev[t,s] = r

     

    print(prob)

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

    print(path)

     

     

     

     

     

     

     

    print("==== simul ====")

    #Vi laver lige transposed, det virker til at give bedre resultat

    p = p.T

    print(p)

    eigval,eigvec = np.linalg.eig(p)

     

    print(eigval)

    print(eigvec)

     

     

     

    #Lav noget simul også...

    Nt = 1000

    st = np.zeros(Nt)

    ot = np.zeros(Nt)

     

    for nt in range(Nt-1):

        RNG = np.random.uniform(0,1)

        #If we are in state 0 then

        if st[nt] == 0:

            if RNG <= p[0,0]:

                st[nt+1] = 0

            elif RNG < p[0,0]+p[1,0]:

                st[nt+1] = 1

            else:

                st[nt+1] = 2

        elif st[nt] == 1:

            if RNG <= p[0,1]:

                st[nt+1] = 0

            elif RNG <= p[0,1]+p[1,1]:

                st[nt+1] = 1

            else:

                st[nt+1] = 2

        else:

            if RNG <= p[0,2]:

                st[nt+1] = 0

            elif RNG <= p[0,2]+p[1,2]:

                st[nt+1] = 1

            else:

                st[nt+1] = 2

       # pass

     

    print(st)

     

    #Her simulate emission observed states

     

    for nt in range(Nt):

        RNG = np.random.uniform()

     

        if st[nt] == 0:

            if RNG <= e[0,0]:

                ot[nt] = 0 #red

            else:

                ot[nt] = 1 #blue

        elif st[nt] == 1:

            if RNG <= e[1,0]:

                ot[nt] = 0

            else:

                ot[nt] = 1

        else:

            if RNG <= e[2,0]:

                ot[nt] = 0

            else:

                ot[nt] = 1

    print(ot)

 


        
        
    fig,ax = plt.subplots()
    plt.title("Hidden states [0,1,2]")
    plt.plot(st[:100])
    #plt.plot(xline[:200][indx0],S[:200][indx0])
    #plt.plot(xline[:200][indx1],S[:200][indx1])
    #plt.plot(xline[:200][indx2],S[:200][indx2])
    plt.xlabel("t")
    plt.ylabel("state")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0,3])
    plt.savefig("3statesimulation.png")
    plt.show()
    
    
    ind0= ot[:100] == 0
    ind1 = ot[:100] == 1
    #o_b = 
    #o_r = ot == 0
    indx = np.array([i for i in range(100)])
    fig,ax = plt.subplots()
    plt.title("Observable states [0,1]")
    
    #plt.plot(ot[:100])
    #plt.plot(xline[:200][indx0],S[:200][indx0])
    #plt.plot(xline[:200][indx1],S[:200][indx1])
    #plt.plot(xline[:200][indx2],S[:200][indx2])
    plt.scatter(indx[ind0],ot[:100][ind0],color="red")
    plt.scatter(indx[ind1],ot[:100][ind1],color="blue")
    plt.xlabel("t")
    plt.ylabel("state")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0,3])
    plt.savefig("2observablestatessimulation.png")
    plt.show()
    
    
    
    count0 = np.sum(st == 0)
    count1 = np.sum(st == 1)
    count2 = np.sum(st == 2)

    #print(st == 2)
    print(count0/Nt,count1/Nt,count2/Nt)
    countvec = np.array([count0/Nt,count1/Nt,count2/Nt])
    print(countvec)
    print(countvec/np.linalg.norm(countvec))
    
    
    import networkx as nx

   
     

    fig,ax = plt.subplots(figsize=(20,20))

    #nx.draw(DG, with_labels=True, font_weight='bold',ax=ax,connectionstyle='arc3, rad = 0.1')


    DG = nx.DiGraph()
    DG.add_edge(2, 1,weight=p[1,0])   # adds the nodes in order 2, 1
    DG.add_edge(1, 2,weight=p[0,1])
    DG.add_edge(1,3,weight=p[0,2])
    DG.add_edge(3,1,weight=p[2,0])
    DG.add_edge(2,3,weight=p[1,2])
    DG.add_edge(3,2,weight=p[2,1])
    DG.add_edge(1, 1,weight=p[0,0])
    DG.add_edge(2, 2,weight=p[1,1])
    DG.add_edge(3, 3,weight=p[2,2])
    pos = nx.spring_layout(DG)
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
                                 label_pos=0.3,
                                 verticalalignment="top",
                                 horizontalalignment="left",bbox = bboxlbl,font_size=40)

    plt.tight_layout()
    plt.savefig("pgraph.png")
    plt.show()


    fig,ax = plt.subplots(figsize=(20,20))

    #nx.draw(DG, with_labels=True, font_weight='bold',ax=ax,connectionstyle='arc3, rad = 0.1')


    DG = nx.DiGraph()
    DG.add_node(1,pos=(1,1),color=None)
    DG.add_node(2,pos=(1,2),color=None)
    DG.add_node(3,pos=(1,3),color=None)
    DG.add_node("R",color="red",pos=(2,1.5))
    DG.add_node("B",color="blue",pos=(2,2.5))
    DG.add_edge(1, "R",weight=round(e[0,0],2),label_pos=0.3)   # adds the nodes in order 2, 1
    DG.add_edge(1, "B",weight=round(e[0,1],2),label_pos=0.3)
    DG.add_edge(2, "R",weight=round(e[1,0],2),label_pos=0.3)   # adds the nodes in order 2, 1
    DG.add_edge(2, "B",weight=round(e[1,1],2),label_pos=0.3)
    DG.add_edge(3, "R",weight=round(e[2,0],2),label_pos=0.3)   # adds the nodes in order 2, 1
    DG.add_edge(3, "B",weight=round(e[2,1],2),label_pos=0.3)

    
    color_map = ["white","white","white","red","blue"]
    #color_map = ['red' if node == "R" for node in G elif node == "B" else 'green']  
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
    
    nx.draw_networkx_nodes(DG, pos,node_color=color_map,node_size=5000)#, nodelist=[0, 1, 2, 3], node_color="tab:red", **options)
    #nx.draw_networkx_nodes(DG, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)
    #e = nx.draw_networkx_edges(DG, pos = pos)#, edge_color='g', arrowsize=50, node_size = 800)
    nx.draw_networkx_edge_labels(DG,pos=pos, edge_labels=labels,connectionstyle='arc3, rad = 0.1',ax=ax,
                                 label_pos=0.3,
                                 verticalalignment="top",
                                 horizontalalignment="left",bbox = bboxlbl,font_size=40)

    plt.tight_layout()
    plt.axis("off")
    plt.savefig("egraph.png")
    plt.show()
     
     
     
     
     
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
    bboxb=dict(facecolor='blue', edgecolor='black', boxstyle="circle,pad=0.3",lw=1)
    bboxr=dict(facecolor='red', edgecolor='black', boxstyle="circle,pad=0.3",lw=1)
    
    yranges = [0, 0.5,1]
    newpath = np.zeros(Npath)
    for k in range(Npath):
        if path[k] == 0:
            newpath[k] = yranges[0]
        elif path[k] == 1:
            newpath[k] = yranges[1]
        else:
            newpath[k] = yranges[2]
    plt.plot(newpath,color="black")
    
    fontsizenode = 18
    for k in range(Npath):
        if O[k] == 0:
            ax.text(k,2,"R",bbox = bboxr,va="center",ha="center",fontsize=fontsizenode)
        else:
            ax.text(k,2,"B",bbox = bboxb,va="center",ha="center",fontsize=fontsizenode)
    
    dy = 0.5
    
    for k in range(Npath):
        for i in range(S):
            if i == path[k]:
                ax.text(k,yranges[i],f"U$_{i+1}$",bbox = bbox2,va="center",ha="center",fontsize=fontsizenode)
            else:
                ax.text(k,yranges[i],f"U$_{i+1}$",bbox = bbox1,va="center",ha="center",fontsize=fontsizenode,alpha=0.5)
                #ax.text(k,1,"U3",bbox = bbox1)
                
                
    plt.axis([-1,Npath+1,0,3])
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("path"+"".join([str(int(i)) for i in O])+".png")
    plt.show()
     

     

     
