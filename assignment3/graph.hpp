#include <fstream>
#include <sstream>
#include<iostream>
#include<vector>
#include<set>
#include<map>
#include<algorithm>
#include<string.h>
#include<assert.h>
#include<stdio.h>
#include "omp.h"

//using namespace std;

class edge
{
  public:
  int32_t source;
  int32_t destination;
  int32_t weight;

};

class graph
{
  private:
  int32_t nodesTotal;
  int32_t edgesTotal;
  int32_t* edgeLen;
  char* filePath;
  int32_t V;
  int32_t E;
  std::map<int32_t,std::vector<edge>> edges;
  int32_t num_levels;
  int32_t* apr;

  public:
  int32_t* indexofNodes; /* stores prefix sum for outneighbours of a node*/
  int32_t* edgeList; /*stores destination corresponding to edgeNo.
                       required for iteration over out neighbours */
  
  graph(char* file)
  {
    filePath=file;
    nodesTotal=0;
    edgesTotal=0;
  }

  std::map<int,std::vector<edge>> getEdges()
  {
      return edges;
  }

   int* getEdgeLen()
  {
    return edgeLen;
  }

    int num_nodes()
  {
      return nodesTotal+1; //change it to nodesToTal
  }
   int num_edges()
  {
      return edgesTotal;
  }
  int get_level()
  {
    return num_levels;
  }
  int *get_csr()
  {
    return edgeList;
  }
  int *get_offset()
  {
    return indexofNodes;
  }
  int *get_aprArray()
  {
    return apr;
  }


   void parseGraph()
  {
    printf("Parsing Graph...");
    FILE *fp = fopen(filePath, "r");
    // reading the number of levels in the graph
    fscanf(fp, "%d %d %d",&num_levels, &V, &E);
    edgesTotal = E;
    apr = new int32_t[V];
     // Fetching the edges and storing edges in list
     for(int i=0; i<E; i++)
     {
        int32_t source;
        int32_t destination;
        int32_t weightVal;
  
        edge e;
        fscanf(fp,"%d %d",&source, &destination);
         
        if(source>nodesTotal)
          nodesTotal=source;


        if(destination>nodesTotal)
            nodesTotal=destination;

        e.source=source;
        e.destination=destination;
        e.weight=1; // Assuming unweighted graph

        edges[source].push_back(e);
      }
    
    assert(V==(nodesTotal+1));
    // Reading APR values
    for(int i=0; i<V; i++)
    {
      fscanf(fp,"%d",&apr[i]);
    }



   // For each vertex v sort the list edges[v] according to source vertex of edge
    #pragma omp parallel for
     for(int i=0;i<=nodesTotal;i++)
     {
       std::vector<edge>& edgeOfVertex=edges[i];

       sort(edgeOfVertex.begin(),edgeOfVertex.end(),
                            [](const edge& e1,const edge& e2) {
                               if(e1.source!=e2.source)
                                  return e1.source<e2.source;

                                return e1.destination<e2.destination;

                            });
     }


     indexofNodes=new int32_t[nodesTotal+2];
     edgeList=new int32_t[edgesTotal];
     edgeLen=new int32_t[edgesTotal];
     int edge_no=0;


    /* Prefix Sum computation for out neighbours
       Loads indexofNodes and edgeList.
    */
    for(int i=0;i<=nodesTotal;i++) //change to 1-nodesTotal.
    {
      std::vector<edge> edgeofVertex=edges[i];

      indexofNodes[i]=edge_no;

      std::vector<edge>::iterator itr;

      for(itr=edgeofVertex.begin();itr!=edgeofVertex.end();itr++)
      {
        edgeList[edge_no]=(*itr).destination;

        edgeLen[edge_no]=(*itr).weight;
        edge_no++;
      }

    }

    indexofNodes[nodesTotal+1]=edge_no;//change to nodesTotal+1.
    fclose(fp); 
   printf("done\n");
 }








};
