---
layout: post
title:  "Patient Representations from Bulk RNA-seq: From Gene Expression to Foundation Models"
date:   2026-08-30 09:00:00 +0200
permalink: biology/bulk-rna-patient-representations
tags: [Bulk RNA-seq, Transcriptomics, Gene Expression, Deep Learning, Computational Biology]
categories: [Biology]
excerpt: "A curated review of how patients are represented from bulk RNA-seq: gene modules, pathway scores, learned pathway embeddings, graph models and foundation models, and what the benchmarks report about each."

---
{% include katex.html %}

<!-- GENERATED FILE -- DO NOT EDIT.
     Source: latex/bulk-rna/main.tex
     Regenerate with: python3 tools/tex2jekyll.py -->

- [Introduction](#introduction)
- [Bulk gene-expression-based representations](#bulk-gene-expression-based-representations)
  - [Gene expression as the representation](#gene-expression-as-the-representation)
  - [Finding data-driven gene modules with WGCNA](#finding-data-driven-gene-modules-with-wgcna)
  - [Summarizing gene modules and pathways](#summarizing-gene-modules-and-pathways)
  - [Learned pathway embeddings](#learned-pathway-embeddings)
  - [Learned gene&ndash;pathway graph embeddings](#learned-gene-pathway-graph-embeddings)
  - [Pathway crosstalk with BinoX](#pathway-crosstalk-with-binox)
  - [Contextualizing pathway embeddings with Pathformer](#contextualizing-pathway-embeddings-with-pathformer)
  - [Gene-level foundation modeling with BulkFormer](#gene-level-foundation-modeling-with-bulkformer)
- [Epilogue: Do learned representations improve on expression itself?](#epilogue-do-learned-representations-improve-on-expression-itself)
- [References](#references)

# Introduction {#introduction}

Biological patient representations turn measurements from one or more modalities into a form that a predictive model can use. This is often a single vector, although some methods produce a set or sequence of vectors that is later aggregated. A useful representation should retain the biological signal needed for tasks such as predicting prognosis or treatment response.

Constructing such representations is difficult, and many approaches are reasonable. This post is a short, curated literature review of representations derived from bulk gene expression, with particular attention to pathways and gene modules. These structures provide useful biological inductive biases and raise interesting modeling questions.

Although the ideas behind deep learning for bulk RNA are interesting, the available evidence does not show that it consistently outperforms simpler representations such as normalized gene expression or PCA. The problem remains important, however, and progress depends partly on understanding and combining ideas from previous work.

This is the post I wish had been available when I began reading about bulk RNA modeling, and I hope it will be useful to others. A future follow-up may focus on histology.

I have likely missed some valuable studies and ideas; contributions and pull requests are welcome.

# Bulk gene-expression-based representations {#bulk-gene-expression-based-representations}

## Gene expression as the representation {#gene-expression-as-the-representation}

For $$N$$ patients and $$G$$ measured genes, write the gene-expression data as 

$$
X = (x_{ig}) \in \mathbb{R}^{N \times G}.
$$

 Here, $$x_{ig}$$ is the expression of gene $$g$$ in patient $$i$$. The simplest patient representation is therefore the corresponding row $$x_i\in \mathbb{R}^{G}$$ after appropriate preprocessing. For RNA-seq, common choices include CPM or TPM normalization and transformations such as $$\log(\mathrm{TPM}+1)$$.

For example, Golub et al. used DNA-microarray gene-expression profiles as patient representations to distinguish acute myeloid leukemia (AML) from acute lymphoblastic leukemia (ALL) and to construct a predictor for new leukemia samples <a href="#ref-golub1999molecular" class="cite">[1]</a>. Their classifier used preprocessed expression values after selecting an informative subset of genes. The important point is that gene expression itself can already be highly predictive.

This continues well into the current era. A 2024 benchmark of bulk RNA-seq representations found that using the transcript-level expression vector itself ("Identity") or PCA matched or outperformed more complex deep-learning representations for cancer-survival prediction <a href="#ref-gross2024robust" class="cite">[2]</a>. We expand on this at the end of this post.

## Finding data-driven gene modules with WGCNA {#finding-data-driven-gene-modules-with-wgcna}

Weighted Gene Co-expression Network Analysis (WGCNA) groups genes that vary together across patients <a href="#ref-langfelder2008wgcna" class="cite">[8]</a>. It first constructs a soft-thresholded adjacency matrix by raising each pairwise correlation to a power $$\beta$$: 

$$
\begin{aligned}
        \rho_{gh} &= \operatorname{cor}(x_{\cdot g},x_{\cdot h}), \\
        a_{gh} &= \vert \rho_{gh}\vert ^{\beta}.
    \end{aligned}
$$

 Strongly correlated genes receive high edge weights, while weak correlations are suppressed.

WGCNA uses topological overlap to compare shared neighborhoods, then applies hierarchical clustering to identify gene modules.

A key distinction is that WGCNA constructs modules from the cohort itself. An alternative is to use predefined gene sets. MSigDB is a broad collection of annotated gene sets assembled from pathways, ontologies, and published expression signatures <a href="#ref-liberzon2011msigdb" class="cite">[9]</a>. Reactome provides manually curated biological reactions and pathways <a href="#ref-milacic2024reactome" class="cite">[10]</a>, while KEGG provides pathway maps linking genes, molecules, and cellular processes <a href="#ref-kanehisa2000kegg" class="cite">[11]</a>. These standardized gene sets are easier to compare across studies, although they may miss structure specific to a particular cohort.

## Summarizing gene modules and pathways {#summarizing-gene-modules-and-pathways}

### PC1 scores {#pc1-scores}

WGCNA summarizes each module by the first principal component (PC1) of its gene expression matrix. This summary is called the module eigengene. Concatenating the module eigengene scores gives one vector per patient.

One subtlety is that the sign of a PCA score is arbitrary. For a module-expression matrix $$X_m$$, PCA finds the direction 

$$
w_1 = \underset{\lVert w\rVert_2=1}{\operatorname{arg\,max}}
    \;\operatorname{Var}(X_m w).
$$

 Both $$w_1$$ and $$-w_1$$ solve this problem, so the sign of PC1 is arbitrary. WGCNA orients the eigengene to correlate positively with the module's average expression profile. After this orientation, positive and negative scores mean that a patient lies above or below the cohort average along the module's shared expression pattern.

### Single-sample Gene Set Enrichment Analysis (ssGSEA) {#single-sample-gene-set-enrichment-analysis-ssgsea}

Before describing ssGSEA, it is useful to note two properties of PC1 scores.

- PC1 captures variation across patients. Genes may be highly expressed in every patient, but this will not produce high PC1 scores if their expression changes little across the cohort.
- PC1 does not ask whether a pathway's genes rank highly relative to other genes in the same patient. It places each patient along the dominant pattern of variation within the selected genes.

ssGSEA takes a different perspective: rather than asking how a pathway varies across patients, it asks where the pathway's genes appear in each individual patient's ranked transcriptome <a href="#ref-barbie2009systematic" class="cite">[12]</a>.

For one patient, ssGSEA ranks all genes by expression. Walking down this list, it increases a running score when it encounters a pathway gene and decreases the score otherwise.

Let $$D(k)$$ denote the running enrichment score after encountering the gene at position $$k$$ in the ranked list. It is updated as 

$$
D(k) = D(k-1) + \Delta_k,
$$

 where, intuitively, 

$$
\Delta_k =
    \begin{cases}
        +\text{rank-weighted contribution}, & \text{if the gene belongs to the pathway}, \\[4pt]
        -\text{constant contribution}, & \text{otherwise}.
    \end{cases}
$$

 More precisely, for a pathway gene set $$S$$, 

$$
\Delta_k =
    \begin{cases}
        \displaystyle
        \frac{r_k^{\alpha}}
        {\sum_{j:g_j \in S} r_j^{\alpha}},
        & g_k \in S, \\[10pt]
        \displaystyle
        -\frac{1}{G-\vert S\vert },
        & g_k \notin S,
    \end{cases}
$$

 where $$G$$ is the total number of genes, $$\vert S\vert $$ is the number of genes in the pathway, $$r_k$$ is a positive weight derived from the rank of gene $$g_k$$, and $$\alpha$$ controls how strongly the highest-ranked genes are weighted. The ssGSEA score sums the running difference across the full ranking: 

$$
\operatorname{ES}_{\mathrm{ssGSEA}}(S) = \sum_{k=1}^{G} D(k).
$$

 Thus PC1 is a cohort-level summary of how a set of genes varies together, whereas ssGSEA is computed within one patient and measures where a pathway's genes fall in that patient's ranked transcriptome.

These summaries are worth considering for two reasons. First, they are strong, interpretable baselines for bulk RNA-seq and are easy to overlook when comparing modern architectures. Second, the same ideas can be generalized into learned models, as the following sections illustrate.

## Learned pathway embeddings {#learned-pathway-embeddings}

Instead of summarizing a pathway with a fixed statistic, we can learn a pathway-specific representation. SurvPath provides one example <a href="#ref-jaume2024modeling" class="cite">[13]</a>. For pathway $$\mathcal{P}_m$$, let $$x_{\mathcal{P}_m}\in\mathbb{R}^{\vert \mathcal{P}_m\vert }$$ contain the expression values of its member genes. The pathway embedding is 

$$
z_m = \phi_m\!\left(x_{\mathcal{P}_m}\right) \in \mathbb{R}^{d},
$$

 where $$\phi_m$$ is a two-layer neural network specific to pathway $$\mathcal{P}_m$$. Because pathways contain different numbers of genes, these networks have different input dimensions and separate parameters, but all produce embeddings of the same dimension $$d$$.

SurvPath stacks these pathway embeddings into a sequence and combines them with histology-patch embeddings using a multimodal Transformer. The pathway encoders and Transformer are trained end to end for survival prediction. This encourages each pathway embedding to retain task-relevant information, which the Transformer aggregates into a patient-level prediction.

## Learned gene&ndash;pathway graph embeddings {#learned-gene-pathway-graph-embeddings}

The independent encoders above do not account for the fact that pathways overlap. A gene may belong to several pathways, but each copy is processed in isolation. ProtoPathway is a recent work that instead represents genes and pathways as the two node types of a bipartite graph <a href="#ref-gallaghersyed2026protopathway" class="cite">[14]</a>. A gene $$g_i$$ is connected to a pathway $$p_j$$ exactly when it belongs to that pathway: 

$$
(g_i,p_j)\in E \quad\Longleftrightarrow\quad g_i\in p_j.
$$

 For each patient, gene nodes are initialized with their expression values and pathway nodes with zeros.

The first layers use GraphSAGE with mean aggregation <a href="#ref-hamilton2017inductive" class="cite">[16]</a>. In simplified form, 

$$
m_v^{(l)} = \frac{1}{\vert \mathcal{N}(v)\vert }
    \sum_{u\in\mathcal{N}(v)}h_u^{(l)},
    \qquad
    h_v^{(l+1)} = \sigma\!\left(
    W^{(l)}[h_v^{(l)}\mathbin\Vert m_v^{(l)}]
    \right).
$$

 Messages travel in both directions. A gene shared by two pathways therefore provides a route through which those pathways can exchange information. The resulting pathway vectors describe both their member genes and their context in the wider pathway graph. Unlike the independent encoders above, the GNN shares parameters across all pathways.

A final GATv2 layer learns how much each neighboring gene should contribute to a pathway embedding: 

$$
z_p = \sum_{g\in\mathcal{N}(p)} \alpha_{gp} W h_g.
$$

 GATv2 makes these weights depend on the pathway doing the querying <a href="#ref-brody2022attentive" class="cite">[15]</a>. A gene shared by two pathways can consequently have different weights in each, so $$\alpha_{g,p_1}$$ need not equal $$\alpha_{g,p_2}$$. ProtoPathway therefore retains named pathway representations while allowing overlapping pathways to contextualize one another through their shared genes.

## Pathway crosstalk with BinoX {#pathway-crosstalk-with-binox}

One useful idea is to estimate pathway-to-pathway relationships from a gene-level interaction network. BinoX provides such a prior for the Pathformer attention mechanism described next.

BinoX estimates whether two pathways are more functionally connected than expected by chance <a href="#ref-ogris2017binox" class="cite">[17]</a>. It uses a functional association network such as FunCoup <a href="#ref-schmitt2014funcoup" class="cite">[18]</a>. Its evidence comes mainly from public databases and experimental datasets. FunCoup combined evidence from interactions, expression, regulation, cellular localization, protein domains, and evolutionary relationships. Many of these resources curate published experiments, so literature contributes indirectly. FunCoup scores each evidence type and combines the scores with a Bayesian model. Its human network contained 18,113 genes and about 4.48 million scored associations. These edges indicate functional evidence, not necessarily direct physical interactions.

For two pathways $$P_A$$ and $$P_B$$, BinoX counts the FunCoup edges connecting their genes: 

$$
k = \sum_{i\in P_A}\sum_{j\in P_B}x_{ij},
$$

 where $$x_{ij}=1$$ if genes $$i$$ and $$j$$ are connected and $$x_{ij}=0$$ otherwise. Large pathways and pathways containing hub genes naturally have more edges, so BinoX compares $$k$$ with degree-preserving randomized networks. It models the null count and tests its upper tail: 

$$
K\sim\operatorname{Binomial}(n,p),
    \qquad
    p_{\mathrm{cross}}=\Pr(K\geq k_{\mathrm{observed}}),
$$

 where $$p$$ is estimated from the randomized networks. A small $$p_{\mathrm{cross}}$$ indicates more pathway crosstalk than expected from the background network.

Applying BinoX to every pathway pair produces a pathway-by-pathway crosstalk matrix. This is fixed external biological knowledge rather than a patient-specific quantity. The next section explains how Pathformer uses it.

## Contextualizing pathway embeddings with Pathformer {#contextualizing-pathway-embeddings-with-pathformer}

Pathformer <a href="#ref-liu2024pathformer" class="cite">[19]</a> creates a patient representation from learned pathway embeddings. The full model uses "criss-cross" attention over both pathways and omics modalities; to stay focused on bulk gene expression, we omit the modality axis here.

First, a single pathway-based sparse neural network uses pathway membership to restrict the allowed gene-to-pathway connections. In the RNA-only configuration, it produces one scalar activity for each pathway: 

$$
s_{ip}=f_{\theta,p}\!\left(x_{i,P_p}\right)\in\mathbb{R},
$$

 where $$x_{i,P_p}$$ contains patient $$i$$'s expression values for genes assigned to pathway $$P_p$$. The scalar is then multiplied by a learned pathway-specific vector $$e_p\in\mathbb{R}^{d}$$ to form the Transformer token: 

$$
z_{ip}=s_{ip}e_p\in\mathbb{R}^{d}.
$$

 Thus the sparse network supplies the patient-specific pathway magnitude, whereas $$e_p$$ supplies a learned pathway identity and direction in embedding space. The multiplication is a particular, restrictive design choice rather than a necessity. A natural alternative would be a pathway-specific MLP that directly outputs a vector, 

$$
z_{ip}=\phi_p\!\left(x_{i,P_p}\right)\in\mathbb{R}^{d},
$$

 as in SurvPath. Whether Pathformer's scalar-times-vector construction provides useful regularization or discards useful within-pathway information is an interesting modeling question that could be verified by ablations. In the full multi-omics configuration, the embedding step can instead be bypassed and the pathway-by-modality values are passed directly to criss-cross attention. Pathformer uses 1,497 pathways curated from KEGG, Reactome, PID, and BioCarta.

The BinoX analysis supplies Pathformer's initial pathway adjacency matrix, denoted in the paper by $$P^{(0)}\in\mathbb{R}^{N_p\times N_p}$$, where $$N_p=1497$$. To distinguish this matrix from the pathway sets, we write 

$$
B_{\mathrm{BinoX}} = P^{(0)},
    \qquad
    (B_{\mathrm{BinoX}})_{ij}
    = s_{\mathrm{BinoX}}(P_i,P_j),
$$

 where $$s_{\mathrm{BinoX}}(P_i,P_j)$$ is the BinoX-derived adjacency value for pathways $$P_i$$ and $$P_j$$. This initial matrix is fixed before training and is the same for every patient <a href="#ref-liu2024pathformer" class="cite">[19]</a>.

The pathway embeddings are stacked into a matrix $$Z$$. In the first Transformer block, attention combines patient-specific pathway similarity with the fixed BinoX prior: 

$$
Q=ZW_Q,\qquad K=ZW_K,\qquad V=ZW_V,
$$

 

$$
A^{(0)}=\operatorname{softmax}\!\left(
        \frac{QK^{\top}}{\sqrt{d}}+B_{\mathrm{BinoX}}
    \right),
    \qquad
    \widetilde{Z}^{(1)}=A^{(0)}V.
$$

 Here, $$QK^{\top}$$ is computed from the current patient's pathway embeddings, whereas $$B_{\mathrm{BinoX}}$$ supplies external biological knowledge. Because it is added to the attention logits, BinoX acts as a bias rather than a hard mask.

In later blocks, the crosstalk matrix is updated from similarities between that patient's contextualized pathway embeddings: 

$$
B_{\mathrm{BinoX}}
    \longrightarrow \text{Block 1}
    \longrightarrow B^{(1)}
    \longrightarrow \text{Block 2}
    \longrightarrow B^{(2)}
    \longrightarrow \text{Block 3}.
$$

 Thus $$B^{(1)}$$ and later matrices are sample-specific; they are not correlations computed across patients or batches. At these later stages, both $$QK^{\top}$$ and $$B^{(l)}$$ describe relationships between learned pathway embeddings. The most distinctive role of the crosstalk term is therefore in the first block, where BinoX introduces information that did not come from the patient data.

Pathformer uses biological knowledge twice: pathway membership constrains the gene-to-pathway encoder, and BinoX initializes pathway-to-pathway attention. It was evaluated on TCGA survival-risk, cancer-stage, and drug-response classification tasks using multi-omics data.

## Gene-level foundation modeling with BulkFormer {#gene-level-foundation-modeling-with-bulkformer}

The methods above explicitly construct pathway representations. BulkFormer instead operates directly on genes and uses a gene&ndash;gene graph as biological prior knowledge <a href="#ref-kang2026bulkformer" class="cite">[21]</a>. It covers 20,010 protein-coding genes and was pretrained on 581,503 human bulk RNA-seq profiles.

BulkFormer constructs one input token for each gene. Its three $$d$$-dimensional components are combined by element-wise addition, not concatenation: 

$$
h_{ig}^{(0)}
    = e_g^{\mathrm{ESM2}}
    + e_{\mathrm{REE}}(x_{ig}^{\mathrm{in}})
    + e_i^{\mathrm{sample}},
    \qquad
    e_i^{\mathrm{sample}}
    = \operatorname{MLP}(x_i^{\mathrm{in}}).
$$

 Here, $$e_g^{\mathrm{ESM2}}$$ represents gene identity using a precomputed embedding of the gene's encoded protein <a href="#ref-lin2023evolutionary" class="cite">[20]</a>, rather than a freely learned gene-ID lookup. The second term represents that gene's continuous expression value. The final term is a compressed summary of the sample-wide input expression vector, and the same vector is added to every gene token. Thus each token contains the gene's identity, its own expression, and global context from the sample. During masked pretraining, $$x_i^{\mathrm{in}}$$ is the partially masked expression profile, so the sample embedding does not directly include the held-out values.

BulkFormer encodes the continuous expression component with a Rotary Expression Embedding (REE) <a href="#ref-kang2026bulkformer" class="cite">[21]</a>. It maps an expression value $$x_g$$ across several frequencies: 

$$
e_{\mathrm{REE}}(x_g)
    = \bigl(
        \sin(\omega_1x_g),\cos(\omega_1x_g),\ldots,
        \sin(\omega_{d/2}x_g),\cos(\omega_{d/2}x_g)
      \bigr).
$$

 This deterministic, multi-frequency encoding preserves the continuity of gene expression without first placing values into discrete bins. It adapts the sinusoidal idea behind rotary position encodings, but uses expression magnitude rather than token position. This is an interesting design choice rather than a necessity: a small MLP could also map each scalar expression value to a vector. REE instead supplies a fixed, smooth basis at several scales, so nearby expression values have related representations without requiring the model to learn that structure from scratch. An MLP would be more flexible, but would introduce additional parameters and could learn an irregular encoding, particularly in expression ranges that are sparsely represented during training.

BulkFormer uses a hybrid GCN&ndash;Performer architecture. The GCN propagates information along a predefined gene graph: 

$$
H^{(l+1)} = \sigma\!\left(
        \widetilde{D}^{-1/2}\widetilde{A}
        \widetilde{D}^{-1/2}H^{(l)}W^{(l)}
    \right).
$$

 Equivalently, the update for gene $$g$$ is 

$$
h_g^{(l+1)} = \sigma\!\left(
        \sum_{u\in\mathcal{N}(g)\cup\{g\}}
        \frac{\widetilde{a}_{gu}}
        {\sqrt{\widetilde{d}_g\widetilde{d}_u}}
        h_u^{(l)}W^{(l)}
    \right),
$$

 where $$\widetilde{a}_{gu}$$ is an entry of the adjacency matrix with self-loops and $$\widetilde{d}_g$$ is its corresponding degree. Thus each gene receives a degree-normalized weighted sum of its own representation and those of its graph neighbors. The self-loop preserves the gene's own identity and expression instead of replacing them with neighboring information. Degree normalization is especially useful in biological graphs, where hub genes can have many neighbors and would otherwise dominate the update.

The authors compared protein-interaction, Gene Ontology, and gene co-expression graphs. The co-expression graph performed best and was used in the final model. It connects each gene to at most its 20 strongest neighbors by absolute Pearson correlation, discarding correlations below 0.4.

The Performer component lets genes exchange information globally. Ordinary self-attention constructs a $$G\times G$$ matrix and therefore scales quadratically with the number of genes: 

$$
\operatorname{Attention}(Q,K,V)
    = \operatorname{softmax}\!\left(
        \frac{QK^{\top}}{\sqrt{d}}
    \right)V.
$$

 Performer approximates the softmax kernel using a random feature map $$\phi$$, 

$$
\exp\!\left(\frac{q^{\top}k}{\sqrt{d}}\right)
    \approx\phi(q)^{\top}\phi(k),
$$

 which allows attention to be rearranged as 

$$
\operatorname{Attention}(Q,K,V)
    \approx D^{-1}\phi(Q)\bigl(\phi(K)^{\top}V\bigr).
$$

 Here, $$D$$ supplies the corresponding normalization. This avoids explicitly constructing the $$G\times G$$ attention matrix and gives approximately linear scaling in the number of genes <a href="#ref-choromanski2021rethinking" class="cite">[22]</a>.

BulkFormer is pretrained by masking 15% of expression values and reconstructing their continuous values from the remaining transcriptome: 

$$
\mathcal{L}_{\mathrm{mask}}
    = \frac{1}{\vert \mathcal{M}\vert }
    \sum_{g\in\mathcal{M}}(x_g-\widehat{x}_g)^2.
$$

 Unlike the pathway-based methods above, BulkFormer learns contextualized gene representations rather than explicit pathway representations. A natural extension would add pathway tokens or a gene&ndash;pathway graph, combining large-scale self-supervised pretraining with named pathway representations.

# Epilogue: Do learned representations improve on expression itself? {#epilogue-do-learned-representations-improve-on-expression-itself}

The evidence is mixed and strongly task-dependent. Across several standardized benchmarks, deep representations have not consistently improved on normalized expression or PCA, particularly for survival prediction in modest-sized patient cohorts. This does not imply that deep learning cannot help; rather, it shows that additional model complexity is not itself sufficient. [Table 1](#table-1) summarizes representative evaluations.

The final column records whether a study reported a direct advantage over an expression or PCA baseline under its own evaluation protocol. The papers use different endpoints and statistical criteria, so the symbols should not be read as a common meta-analytic significance test. A green check indicates a clear direct gain, a red cross indicates that expression or PCA was on par or better, an orange triangle indicates a method- or setting-dependent gain, and a dash indicates that a matched comparison was not reported. This distinction matters for Pathformer and BulkFormer: both improved on their selected baselines, but neither paper established superiority over a matched RNA-only expression or PCA baseline across its tasks.

<div class="table-scroll">
<table id="table-1" class="eval-table">
<thead><tr><th>Study</th><th>Dataset</th><th>Task</th><th>Modeling method</th><th>Direct gain?</th></tr></thead>
<tbody>
<tr><td>Smith et al. <a href="#ref-smith2020standard" class="cite">[3]</a></td><td>${\sim}45{,}000$ recount2 samples</td><td>24 classification; 26 survival tasks</td><td>No embedding, PCA, SDAE, and VAE; matched downstream predictors</td><td class="eval-cell"><span class="eval-no">&#215;</span></td></tr>
<tr><td>Way and Greene <a href="#ref-way2017evaluating" class="cite">[4]</a></td><td>${>}10{,}000$ TCGA tumours</td><td>NF1-inactivation prediction; HGSC subtype analysis</td><td>Three VAEs versus PCA, ICA, NMF, and ADAGE</td><td class="eval-cell"><span class="eval-no">&#215;</span></td></tr>
<tr><td>Gross et al. <a href="#ref-gross2024robust" class="cite">[2]</a></td><td>TCGA: 11 cohorts and 33-cancer pan-cancer data</td><td>Censoring-aware survival</td><td>AE variants, VAE, MAE, and GNN versus Identity and PCA</td><td class="eval-cell"><span class="eval-no">&#215;</span></td></tr>
<tr><td>Gross et al. <a href="#ref-gross2024robust" class="cite">[2]</a></td><td>DepMap cell lines</td><td>Gene-essentiality prediction</td><td>The same representation families</td><td class="eval-cell"><span class="eval-partial">&#9651;</span></td></tr>
<tr><td>Pathformer <a href="#ref-liu2024pathformer" class="cite">[19]</a></td><td>TCGA multi-omics cohorts</td><td>Risk group, stage, and drug response</td><td>Pathway-informed Transformer versus 18 integration methods</td><td class="eval-cell"><span class="eval-none">&mdash;</span></td></tr>
<tr><td>GexBERT <a href="#ref-jiang2025gexbert" class="cite">[5]</a></td><td>TCGA; 14 cancer-specific survival tasks</td><td>Survival representation</td><td>Transformer summary embedding versus 200-component PCA</td><td class="eval-cell"><span class="eval-yes">&#10003;</span></td></tr>
<tr><td>GexBERT <a href="#ref-jiang2025gexbert" class="cite">[5]</a></td><td>TCGA; 64&ndash;1,024 randomly selected input genes</td><td>Survival from incomplete panels</td><td>Observed plus restored anchor genes versus observed genes alone</td><td class="eval-cell"><span class="eval-partial">&#9651;</span></td></tr>
<tr><td>Xia et al. <a href="#ref-xia2022crossstudy" class="cite">[6]</a></td><td>NCI-60, CTRP, GDSC, CCLE, and gCSI</td><td>Cross-study drug response</td><td>RF, LightGBM, and deep neural networks, including UnoMT</td><td class="eval-cell"><span class="eval-none">&mdash;</span></td></tr>
<tr><td>SurvBoard <a href="#ref-wissel2025survboard" class="cite">[7]</a></td><td>28 cohorts from TCGA, ICGC, TARGET, and METABRIC</td><td>Gene-expression-only survival</td><td>Elastic net and survival RF versus neural-network models</td><td class="eval-cell"><span class="eval-no">&#215;</span></td></tr>
<tr><td>BulkFormer <a href="#ref-kang2026bulkformer" class="cite">[21]</a></td><td>${>}500{,}000$ training profiles; multiple bulk RNA-seq benchmarks</td><td>Five downstream tasks</td><td>BulkFormer versus seven single-cell foundation models transferred to bulk data</td><td class="eval-cell"><span class="eval-none">&mdash;</span></td></tr>
</tbody></table></div>
<p class="table-caption"><strong>Table 1.</strong> Representative evaluations of bulk-transcriptomic modeling. Results are not directly comparable across rows.</p>

<span class="eval-yes">&#10003;</span> reported direct gain; <span class="eval-no">&#215;</span> expression or PCA on par or better; <span class="eval-partial">&#9651;</span> gain limited to some methods or input regimes; <span class="eval-none">&mdash;</span> no matched expression/PCA comparison.

For Gross et al., deep representations had a slight, consistent advantage on the DepMap task overall, but some were equivalent rather than superior to the baselines under the paper's pairwise acceptance criterion <a href="#ref-gross2024robust" class="cite">[2]</a>.

For GexBERT, the green check records the numerical improvement of its summary embedding over PCA in the reported survival results, rather than a standardized significance test. Reconstructed expression was useful mainly for small, randomly selected panels; its benefit diminished for larger panels and was negligible when the observed genes were already selected for prognostic relevance <a href="#ref-jiang2025gexbert" class="cite">[5]</a>.

The open research problem is therefore not simply to build a more expressive model. A useful representation must retain patient-specific gene information, exploit sufficiently large training cohorts, and introduce biological structure without discarding predictive signal. Pathways and gene modules may provide useful regularization and interpretability; hybrid representations that preserve gene-level information remain a promising direction.

# References {#references}

<ol class="references">
<li id="ref-golub1999molecular">T. R. Golub, D. K. Slonim, P. Tamayo, et al.
 Molecular classification of cancer: Class discovery and class
prediction by gene expression monitoring.
 <em>Science</em>, 286(5439):531&ndash;537, 1999.
 doi:10.1126/science.286.5439.531.</li>
<li id="ref-gross2024robust">B. Gross, A. Dauvin, V. Cabeli, et al.
 Robust evaluation of deep learning-based representation methods for
survival and gene essentiality prediction on bulk RNA-seq data.
 <em>Scientific Reports</em>, 14:17064, 2024.
 doi:10.1038/s41598-024-67023-8.</li>
<li id="ref-smith2020standard">A. M. Smith, J. R. Walsh, J. Long, et al.
 Standard machine learning approaches outperform deep representation
learning on phenotype prediction from transcriptomics data.
 <em>BMC Bioinformatics</em>, 21:119, 2020.
 doi:10.1186/s12859-020-3427-8.</li>
<li id="ref-way2017evaluating">G. P. Way and C. S. Greene.
 Evaluating deep variational autoencoders trained on pan-cancer gene
expression.
 <em>arXiv preprint arXiv:1711.04828</em>, 2017.</li>
<li id="ref-jiang2025gexbert">S. Jiang and S. Hassanpour.
 Transformer-based representation learning for robust gene expression
modeling and cancer prognosis.
 <em>Scientific Reports</em>, 15:37581, 2025.
 doi:10.1038/s41598-025-14949-2.</li>
<li id="ref-xia2022crossstudy">F. Xia, J. E. Allen, P. Balaprakash, et al.
 A cross-study analysis of drug response prediction in cancer cell
lines.
 <em>Briefings in Bioinformatics</em>, 23(1):bbab356, 2022.
 doi:10.1093/bib/bbab356.</li>
<li id="ref-wissel2025survboard">D. Wissel, N. Janakarajan, A. Grover, E. Toniato, M. Rodriguez Martinez,
and V. Boeva.
 SurvBoard: Standardized benchmarking for multi-omics cancer survival
models.
 <em>Briefings in Bioinformatics</em>, 26(5):bbaf521, 2025.
 doi:10.1093/bib/bbaf521.</li>
<li id="ref-langfelder2008wgcna">P. Langfelder and S. Horvath.
 WGCNA: An R package for weighted correlation network analysis.
 <em>BMC Bioinformatics</em>, 9:559, 2008.
 doi:10.1186/1471-2105-9-559.</li>
<li id="ref-liberzon2011msigdb">A. Liberzon, A. Subramanian, R. Pinchback, H. Thorvaldsdottir, P. Tamayo, and
J. P. Mesirov.
 Molecular signatures database (MSigDB) 3.0.
 <em>Bioinformatics</em>, 27(12):1739&ndash;1740, 2011.
 doi:10.1093/bioinformatics/btr260.</li>
<li id="ref-milacic2024reactome">M. Milacic, D. Beavers, P. Conley, et al.
 The Reactome Pathway Knowledgebase 2024.
 <em>Nucleic Acids Research</em>, 52(D1):D672&ndash;D678, 2024.
 doi:10.1093/nar/gkad1025.</li>
<li id="ref-kanehisa2000kegg">M. Kanehisa and S. Goto.
 KEGG: Kyoto Encyclopedia of Genes and Genomes.
 <em>Nucleic Acids Research</em>, 28(1):27&ndash;30, 2000.
 doi:10.1093/nar/28.1.27.</li>
<li id="ref-barbie2009systematic">D. A. Barbie, P. Tamayo, J. S. Boehm, et al.
 Systematic RNA interference reveals that oncogenic KRAS-driven
cancers require TBK1.
 <em>Nature</em>, 462:108&ndash;112, 2009.</li>
<li id="ref-jaume2024modeling">G. Jaume, A. Vaidya, R. J. Chen, D. F. K. Williamson, P. P. Liang, and
F. Mahmood.
 Modeling dense multimodal interactions between biological pathways
and histology for survival prediction.
 In <em>CVPR</em>, pages 11579&ndash;11590, 2024.
 doi:10.1109/CVPR52733.2024.01100.</li>
<li id="ref-gallaghersyed2026protopathway">A. Gallagher-Syed, C. Pitzalis, M. J. Lewis, M. R. Barnes, and G. Slabaugh.
 ProtoPathway: Biologically structured prototype-pathway fusion for
multimodal cancer survival prediction.
 <em>arXiv preprint arXiv:2605.21454</em>, 2026.</li>
<li id="ref-brody2022attentive">S. Brody, U. Alon, and E. Yahav.
 How attentive are graph attention networks?
 In <em>International Conference on Learning Representations
(ICLR)</em>, 2022.</li>
<li id="ref-hamilton2017inductive">W. L. Hamilton, R. Ying, and J. Leskovec.
 Inductive representation learning on large graphs.
 In <em>Advances in Neural Information Processing Systems</em>,
pages 1024&ndash;1034, 2017.</li>
<li id="ref-ogris2017binox">C. Ogris, D. Guala, T. Helleday, and E. L. L. Sonnhammer.
 A novel method for crosstalk analysis of biological networks:
improving accuracy of pathway annotation.
 <em>Nucleic Acids Research</em>, 45(2):e8, 2017.
 doi:10.1093/nar/gkw849.</li>
<li id="ref-schmitt2014funcoup">T. Schmitt, C. Ogris, and E. L. L. Sonnhammer.
 FunCoup 3.0: Database of genome-wide functional coupling networks.
 <em>Nucleic Acids Research</em>, 42(D1):D380&ndash;D388, 2014.
 doi:10.1093/nar/gkt984.</li>
<li id="ref-liu2024pathformer">X. Liu, Y. Tao, Z. Cai, et al.
 Pathformer: A biological pathway informed transformer for disease
diagnosis and prognosis using multi-omics data.
 <em>Bioinformatics</em>, 40(5):btae316, 2024.
 doi:10.1093/bioinformatics/btae316.</li>
<li id="ref-lin2023evolutionary">Z. Lin, H. Akin, R. Rao, et al.
 Evolutionary-scale prediction of atomic-level protein structure with
a language model.
 <em>Science</em>, 379(6637):1123&ndash;1130, 2023.
 doi:10.1126/science.ade2574.</li>
<li id="ref-kang2026bulkformer">B. Kang, R. Fan, M. Yi, C. Cui, and Q. Cui.
 BulkFormer: A large-scale foundation model for bulk transcriptomes.
 <em>Cell Systems</em>, 17(7):101657, 2026.
 doi:10.1016/j.cels.2026.101657.</li>
<li id="ref-choromanski2021rethinking">K. Choromanski, V. Likhosherstov, D. Dohan, et al.
 Rethinking attention with performers.
 In <em>International Conference on Learning Representations
(ICLR)</em>, 2021.</li>
</ol>
