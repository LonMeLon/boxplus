# boxplus

\subsection{WordNet Hypernym Prediction Task}
The WordNet hypernym hierarchy contains 837k edges after processing the transitive relation on the edges. Positive examples are randomly chosen from the 837k edges, and negative examples are sampled by replacing the term with a random word of the dictionary. We use the same training/testing data as in Li et al. \cite{li2018smoothing}. Baseline models are trained using the same parameters of Li et al. \cite{li2018smoothing}. 
The batch size and the dimension is set to 800 and 50, the regularization weight is set to 0.005, and the AdamW optimizer is adopted with the learning rate of 0.001.

\subsection{WordNet Noun Hierarchy Prediction Task}
The WordNet noun hierarchy contains 82,114 entities and 84,363 edges in its transitive reduction. We use the same training settings in Dasgupta et al. \cite{dasgupta2020improving}, and regularization methods in Patel and Sankar \cite{patel2020representing} for training box embedding-based baselines. We fine-tune the gumbel $\beta \in \left[0.001, 3\right]$, learning rate $\in \left[0.0005, 1\right]$, batch size $\in \left\{512, 1024, 2048, 8096, 16192\right\}$, and regularization weight $\in \left[0.0001, 0.005\right]$. We report the F1 score on the test set of size 28,838 edges by sampling 1 : 10 negative edges, based on the computed conditional probability. 

\subsection{\textbf{Ranking Task on Tree-Structured Data}} 
For the dataset, \textbf{Balanced tree} consists of 40 nodes with branching factor 3 and depth 4. 102 transitive edges were used for training. \textbf{Mammal Hierarchy of WordNet} contains 1,182 entities, 6,542 transitive edges were used for training. MRR is calculated on a subset of size 3,441 edges which are sampled randomly from the transitive data. \textbf{Random tree} has 3,000 nodes, which is generated randomly using networkx. We train only on the transitive reduction of 2,999 edges. The MRR is calculated on 4,920 edges which are randomly sampled from the whole transitive data.
We use the same training settings in Dasgupta et al. \cite{dasgupta2020improving}. We fine-tune the gumbel $\beta \in \left[0.001, 3\right]$, learning rate $\in \left[0.0005, 1\right]$, batch size $\in \left\{256, 512, 1024, 2048, 4096\right\}$, and number of negative samples $\in \left\{2, 5, 10, 20, 25, 40, 70\right\}$.

\subsection{\textbf{Flickr}}
Flickr is an image dataset containing 45 million images. We use the same training settings and dataset split mentioned in Dasgupta et al. \cite{dasgupta2020improving}. We fine-tune the gumbel $\beta \in \left[0.0001, 1\right]$, learning rate $\in \left[1e-5, 0.01\right]$, and batch size $\in \left\{128, 256, 512\right\}$.

\subsection{\textbf{MovieLens}}
In this task, the training/testing data is processed by the following steps: We first collect all pairs of user-movie ratings higher than 4 points from the MovieLens-20M dataset. Then, the dataset is further pruned using the popularity of the movies. Only the movies that have been rated for more than 100 times are retained. We use the same training settings and datasets mentioned in Dasgupta et al. \cite{dasgupta2020improving}. The batch size and the dimension is set to 128 and 50, and the AdamW optimizer is adopted with the learning rate of 0.001. 

\subsection{\textbf{Amazon Books}}
The dataset is filtered so that only the users with at least 20 interactions remain. We use the following recommendation models as our baselines in the experiment. We test the $\beta$ in GumbelBox $\in$ $\left[0.001, 0.01, 0.1\right]$, and choose dimension $d = 128$.
\begin{itemize}
	\item DSSM \cite{huang2013learning} represent users and items based on vector embeddings. The relevance score is calculated based on the dot product between the vector embeddings of user and item.
	\item GRU4REC \cite{hidasi2015session} model user interaction sequence using GRU network to perform session-based recommendation task.
	\item SASREC \cite{kang2018self} utilize multi-head self-attention mechanism to model sequential recommendation.
\end{itemize}

\subsection{\textbf{MSMARCO Passage}}
The dataset contains 8,841,823 passages, 502,939 training queries, and 6,980 test queries. We use the following recommendation models as our baselines in the experiment. For STAR and AR2+SimANS, the inner product is used to calculate the relevance score. Faiss\cite{johnson2019billion} is adopted to efficiently search the relevance passages. For box embedding-based models, we infer the box representations of all passages and queries, and calculate the relevance scores based on our models.
\begin{itemize}
	\item STAR\cite{zhan2021optimizing} utilizes both random sampling negatives and the static hard negatives to train model.
	\item AR2+SimANS\cite{zhou2022simans} adopts ambiguous negative sampling to train model based on the negative sampling distribution from AR2 model \cite{zhang2021adversarial}.
\end{itemize}
