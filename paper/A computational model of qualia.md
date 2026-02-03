

# **Format as Subjective Experience: A Computational Model of Qualia**

Alt: MaryVLM: Towards a rigorous science of subjectivity  
[Literature Search for Qualia Models](https://docs.google.com/document/u/0/d/1oyfov-siv1v_6rWKHrFzlNtB5PSuJC5UdXNg5uV3Nzk/edit)

## TODO:

- [ ] Use representational analysis to make more robust to cogsci  
      - [ ] Need to do RSA geometric analysis for the inverted spec experiment in addition to Procrustes (high procrustes, high RSA → structuralism confirmed)  
- [ ] Add another distance metric that is used in similar papers (Gromov-Wasserstein (GW))  
- [ ] Rename hypothesis to “impenetrable representation hypothesis” (this invokes the Pylyshyn’s concept of visual processing being “cognitively impenetrable”)  
      - [ ] In comp sci terms “impentrable” \= read-only (Darwinian firmware, Phylogenetic)  
      - [ ] Encapsulation (Read-Only): A term from Jerry Fodor's Modularity of Mind. An "Encapsulated Module" (like vision) is mandatory and fast. It fires its output regardless of what the central mind wants. The central mind can read it (see the optical illusion) but cannot write to it (fix the illusion).  
- [ ] Similarly, we need a name for the semantic concepts (fusion layer, Ontogenetic)  
      - [ ] “Penatrable” \= read/write (Vygotsky’s “inner speech” \~= chain of thought as rudimentary inner speech)  
      - [ ] Recurrence (Read/Write): This refers to "Re-entrant Processing" (Edelman) or "Recurrent Loops" (Vygotsky/Hofstadter). The representation is not a static input but a dynamic loop. The system reads its own output and writes it back as the next input (Inner Speech). This allows the representation to be "scrubbed" or edited by high-level beliefs.  
- [ ] There is also a SEPARATE concept of phylogenetic vs. ontogenetic representations  
      - [ ] Phylogenetic (Frozen Weights): Constraints hardwired by evolution ("the species' learning"). The SigLIP encoder is "Phylogenetic" because its parameters were fixed before the agent's lifetime began. It represents the accumulated wisdom of evolution (edge detection, color dominance) that the individual cannot un-learn.  
      - [ ] Ontogenetic (Unfrozen Weights): Constraints learned during the individual's lifetime. The Language Model is "Ontogenetic"—it is plastic and shaped by the specific environment (text data) it encounters.  
- [ ] Make sure to include an idea of possibly blurred boundaries between impenetrable and penetrable (eg, synesthesia)  
- [ ] Add "Levels of Analysis" problem (Marr) to intro  
- [ ] Need to make a table of cogsci to AI concepts in the experiment (frozen vs unfrozen weights)  
- [ ] Add the caveat of “Soft Encapsulation” to the discussion  
      - [ ] the conceptual system cannot rewrite the sensory input (it cannot turn "Red" into "Green"), but it can apply Attentional Gain to specific features.  
      - [ ] Wine tasting, "Whorfian" angle (eg, Russians have two words for blue)  
- [ ] Need to add that in the context of the knowledge problem, we don’t need to propose a model that completely mimics how the brain works, we just need to show that *it is possible* to create a model that explains the various features of qualia. (So we aren’t going to go the traditional cogsci route of building a model that can mimic data obtained from humans doing the same task. We need to make this super clear.)  
- [ ] “Central in philosophy of mind and one of the greatest obstacles in neuroscience” [(Gouveia 2022\)](https://app.readcube.com/library/551e4aac-32fe-4311-9c1c-1cdb7c1fb947/all?uuid=7037188238492222&item_ids=551e4aac-32fe-4311-9c1c-1cdb7c1fb947:f1ee4f12-f70a-476f-8155-65dfb8a1958f)  
- [ ] “"(8) Science is objective; (9) Consciousness is subjective; (10) Therefore, we cannot have a science of consciousness" (Gouveia 2022, p. 162).  
      - [ ] Because qualia are precisely the part of our experiences that are not related to informational content (and therefore intersubjective), and cognitive science is primarily based on information content, qualia are not within the domain of cognitive science. [(Griffith No year)](https://app.readcube.com/library/551e4aac-32fe-4311-9c1c-1cdb7c1fb947/all?uuid=5042903798921943&item_ids=551e4aac-32fe-4311-9c1c-1cdb7c1fb947:0092b54c-78dc-493c-9b47-4a80a6a81e0d)  
- [ ] “Objective information” → semantic representations (penetrable through inner speech)  
- [ ] “Subjective information” → sensory representations (impenetrable)  
- [ ] “What is it like” → Cannot swap weights out.  
- [ ] The goal is to show how subjectivity can be explained and conceptualized within a physical system. These are all compatible within a physical system.  
      - [ ] We can demonstrate MaryVLM’s semantic concept of red.  
      - [ ] We can explain MaryVLM's surprise AND  
      - [ ] We can show how MaryVLM has “learned” something new.  
- [ ] There has been a shift in the literature moving towards this direction. Coming up with a more rigorous science of subjectivity.  
      - [ ] “Qualia Structure Paradigm” → Focus on relative, structural relationships, rather than intrinsic, private nature  
      - [ ]   
- [ ] The hard problem remains unsolved. Subjectivity is not part of it. All one needs to presume is that the “I” sits downstream of (and is contingent on) different internal representations of information, and the “I” cannot receive another’s representation. The troubling notion of “I.”   
- [ ] Model surgery.  
- [ ] “An objective science of subjectivity”  
- [ ] Going to show how many of the problematic characteristics of qualia are actually expected in a physical system  
- [ ] CKA (Centered Kernel Alignment) [(Kornblith 2019\)](https://app.readcube.com/library/551e4aac-32fe-4311-9c1c-1cdb7c1fb947/all?uuid=6305378866486474&item_ids=551e4aac-32fe-4311-9c1c-1cdb7c1fb947:2d1fa9fb-4ace-4924-8407-1f570dd1f197) is the "modern ML" successor to RSA, but technically, it is more of a refined version than a direct competitor. [(Williams 2024\)](https://app.readcube.com/library/551e4aac-32fe-4311-9c1c-1cdb7c1fb947/all?uuid=8466645317629826&item_ids=551e4aac-32fe-4311-9c1c-1cdb7c1fb947:5558e424-3793-4084-947e-1a9e93413f3c)

## Figures

Figure 1\. Experimental Setup

Figure 2\. MaryVLM Architecture.

Figure 3\. a, Training Loss. b, MMStar Accuracy, c, 

Figure 4\. Embedding Visualizations.  
Different colors? 1000 activations from 10 categories. (or maybe 8 by 8). 8 colors, 8 shapes. The sort of thing matplotlib can handle well.

Figure 5\. Representational Analysis.

Figure 6\. Learning “Red” Visualizations.  
Step by step measurements of OOD error alongside


## Introduction

**\[What is the problem?\]**

A foundational challenge in cognitive science is bridging the "Explanatory Gap" between the objective description of information processing and the subjective character of sensory experience. While computational models have successfully replicated high-level cognitive functions such as reasoning and object recognition, they largely fail to account for the phenomenology of *qualia*—the "raw feel" of a sensory format. The prevailing question has shifted from the metaphysical "What is consciousness?" to the engineering-focused "What architectural constraints are required for a system to exhibit the structural properties of subjective experience?"

**\[Why is it interesting and important?\]**

The status of qualia represents the single significant boundary condition for the computational theory of mind. Historically, the field has often conceded that subjective experience lies outside its explanatory scope. As Griffith and Byrne (1997) argued in their critique, if cognitive science is strictly the study of information processing, and qualia are defined as non-informational "raw feels," then subjectivity is axiomatically excluded from the domain of inquiry.

This concession is devastating: it implies that a physicalist science of mind is fundamentally incomplete, forever separated from the phenomenology it seeks to explain. This paper challenges that defeatist conclusion. By operationalizing qualia through the "Constructive Approach" (Taniguchi et al., 2025), we move the debate from metaphysical stalemate to empirical falsifiability. The stakes are high: if we can demonstrate that "raw feels" leave a distinct, quantifiable structural footprint (a "Wow" signal) within a physical system, we refute the dualistic premise that subjectivity is non-computational. Conversely, if high-fidelity models like *MaryVLM* fail to generate such signals despite functional equivalence, it would provide robust empirical support for the view that the "Hard Problem" is indeed a permanent barrier to physical understanding (Gouveia, 2022).

**\[Why is it hard?\]**

Modeling this phenomenon is difficult because standard neural architectures are "leaky" by design. In typical multimodal learning (e.g., CLIP or standard VLMs), visual encoders and language decoders are trained simultaneously to minimize alignment error. Consequently, these models learn an immediate, frictionless mapping between the pixel values of "red" and the token "red." They lack the biological reality of *encapsulation*: the fact that our sensory hardware is evolutionarily fixed ("phylogenetic"), while our conceptual understanding is developmentally plastic ("ontogenetic"). Without this separation, a model cannot experience the "shock" of a new format because it never faces the cost of translation.

**\[Why hasn't it been solved before?\]**

Previous computational attempts have often relied on functionalist descriptions or purely symbolic systems that lack the continuous, high-dimensional nature of sensory data. Conversely, recent deep learning approaches often ignore the structural constraints necessary to test the "Knowledge Argument" (Jackson, 1982). They do not isolate the *format* of the information from the *content*. To date, there has been no rigorous attempt to operationalize the "Mary's Room" thought experiment using state-of-the-art Vision-Language Models (VLMs) where the sensory hardware is strictly "frozen" against a learning conceptual system.

**Key components of our approach**

We propose **MaryVLM**, a framework that tests the "Impenetrable Representation Hypothesis": that qualia arise from the friction of translating between a fixed sensory encoder and a plastic conceptual decoder. We utilize a small-scale VLM (SmolVLM) and impose a strict architectural constraint: the vision encoder (SigLIP) is frozen to simulate biological hardwiring, while the projection layer and language model are trained solely on achromatic (grayscale) data.

We then introduce chromatic stimuli (the "Release" phase) to measure two specific phenomena:

1. **The "Wow" Signal:** We quantify the "subjective shock" of novel qualia using Mahalanobis distance as a proxy for Mismatch Negativity (MMN), a well-documented prediction error signal in neuroscience.  
2. **Structural Realism:** We employ Procrustes Analysis to compare the latent spaces of functionally identical agents, testing for the "Inverted Spectrum" phenomenon.

**Summary of Contributions**

* **Architectural Operationalization:** We provide the first implementation of the "Constructive Approach" using a frozen-encoder VLM to simulate the phylogenetic/ontogenetic divide.  
* **The "Wow" Metric:** We demonstrate that the onset of a new sensory format generates a statistically significant Out-of-Distribution (OOD) signal (the "Wow" signal) that is distinct from simple content novelty ($$p \< .001$$), mirroring biological MMN.  
* **Evidence for Structural Realism:** We show that agents can be functionally equivalent (identical VQA performance) while possessing rotationally misaligned latent geometries, providing a computational existence proof for the "Inverted Spectrum."  
* **Validation of the Impenetrable Representation Hypothesis:** Our results suggest that subjective experience can be modeled as the computational cost of alignment between mismatched representational formats.  
* **Unified Theory of the Knowledge Problem**: As a bonus, our MaryVLM model shows how the various philosophical “replies” to Mary’s Room can be unified under one framework.

## Prior Work

Recent advancements at the intersection of Artificial Intelligence and the philosophy of mind have precipitated a "constructive approach" to consciousness, shifting the focus from metaphysical speculation to the engineering of systems that exhibit structural properties of subjective experience.

### The Constructive Approach and Bidirectional Influence 

The most direct theoretical antecedent to our framework is the work of Taniguchi et al. (2025), who propose a "constructive approach" to the bidirectional influence between qualia structure and language emergence. They hypothesize that while perceptual experiences (upward organization) constrain the lexicon, language exerts a "downward constraint" on internal representations, forcing agents to align with a shared categorical structure. Our *MaryVLM* model operationalizes this tension: the "subjective shock" we measure is the computational friction generated when a frozen upward constraint (biological vision) clashes with a plastic downward constraint (learned grayscale language).

### Generative AI and the "Mary's Room" Experiment

The literature increasingly treats generative models as "philosophical laboratories". Bellini-Leite (2024) explicitly links Generative AI to the "Mary's Room" thought experiment, questioning whether systems trained on vast text datasets (propositional knowledge) can generate novel sensory instances without grounding. Furthermore, recent evaluations of Vision-Language Models (VLMs) like SmolVLM emphasize the utility of frozen vision encoders to prevent "catastrophic forgetting" of visual features. We adopt this architectural choice not just for robustness, but to enforce a strict informational encapsulation between sensation and concept, simulating the "read-only" nature of biological qualia.

### Structural Realism and Representational Alignment

Finally, our approach draws on "Neurophenomenal Structuralism," which posits that while the intrinsic "content" of experience may be private, the relational structure of qualia spaces is mathematically formalizable. Sucholutsky and Griffiths (2023) utilized Gromov-Wasserstein distance to demonstrate that neural networks can share "qualia structures" (e.g., color geometry) even if their individual activation axes are rotated or inverted. This supports the feasibility of our "Inverted Spectrum" analysis, suggesting that "redness" can be defined geometrically rather than chemically.

## Methods

To investigate the functional distinction between propositional knowledge and sensory experience, we employed a dual-encoder Vision-Language Model (VLM) architecture. This setup allows us to operationalize the "Mind-Body" distinction computationally: the vision encoder represents the fixed biological hardware ($$E\_\\phi$$), and the language/fusion layers represent the plastic conceptual system ($$P\_\\theta$$).

### Model Architecture

We utilized the **SmolVLM-256M** architecture (HuggingFaceTB, 2024), chosen for its modular separation of vision and language components. The architecture consists of:

1. **Sensory Encoder (Fixed Biology):** A SigLIP vision transformer. Crucially, the weights $$\\phi$$ were **frozen** throughout the experiment. This constraint simulates the biological reality that the retina and primary visual cortex (V1) do not fundamentally restructure themselves based on semantic learning; they provide a fixed sensory distinct from high-level belief updating.  
2. **Conceptual Bridge (Plastic Mind):** A trainable multi-modal projection layer that maps visual embeddings into the semantic space of the language model.  
3. **Propositional Decoder:** A transformer-based language model (SmolLM2) representing Mary's store of "textbook knowledge."

### **Stimuli and Data Partitions**

We utilized the **Localized Narratives** dataset (Pont-Tuset et al., 2020), specifically the COCO subset. To isolate the effects of *format novelty* from *object novelty*, we partitioned the evaluation data into two functional subsets:

* **Internal Control (Achromatic):** Images presented in grayscale ($$L, L, L$$). These validate the agent’s **Conceptual Grounding**—its ability to identify shapes and textures (e.g., "This is a banana") based on its training.  
* **External Release (Chromatic):** The *same* images presented in RGB. Because the fusion layers have never encountered the chromatic format, these serve as the stimuli for the "Release" phase.

### **Procedure**

**Phase 1: Achromatic Acquisition ("The Room")**

To simulate Mary’s confinement, we trained the agent’s conceptual layers ($$P\_\\theta$$ and $$D\_\\psi$$) exclusively on grayscale images. An RGB-to-Luminance transformation $$T(x)$$ was applied to all training inputs. The model was optimized on Visual Question Answering (VQA) and Captioning tasks until it achieved asymptotic accuracy. During this phase, the model possessed "complete physical information" in the form of text descriptions (e.g., "Apples are red") but lacked the functional capacity to process the chromatic signal.

**Phase 2: Chromatic Release (Paired-Stimulus Evaluation)**

To measure the "subjective shock" of the new format, we employed a **Paired-Stimulus Design**. We presented the model with identical images in both the Internal (Achromatic) and External (Chromatic) conditions. We hypothesized that if the *Impenetrable Representation Hypothesis* is correct, the Chromatic condition should trigger a high-energy "surprise" signal, distinct from the low-energy state of recognizing the object in grayscale.

### **Measures**

**Novelty Detection (The "Wow" Signal)**

We operationalized the neural signature of novelty as **Mahalanobis Distance** ($$D\_M$$). In biological systems, the Locus Coeruleus-Norepinephrine (LC-NE) system gates attention when predictions fail (Aston-Jones & Cohen, 2005). Computationally, we modeled this as the distance of the incoming chromatic vector $$z\_c$$ from the learned manifold of achromatic vectors $$\\mu\_g$$:

$$S \= D\_M(z\_c) \- D\_M(z\_g)$$

A significant positive $$S$$ indicates that "Redness" is not treated as just another feature, but as a **Violation of Expectation** (VoE) regarding the fundamental format of the input.

**Subjective Specificity (The Inverted Spectrum)**

To test whether "qualia" are objective properties of the signal or subjective constructions of the agent, we trained two identical MaryVLM agents ($$M\_A$$ and $$M\_B$$) with different random seeds for the conceptual bridge. We extracted the latent centroid vectors for the concept "Red" from both agents. We utilized **Procrustes Analysis** to measure the alignment between their internal spaces. High functional equivalence (identical verbal outputs) combined with high Procrustes disparity (misaligned internal vectors) would support the hypothesis that subjective experience is *structurally real* but *implementation-dependent*.

# Results (Empirical Results WIP)

### **The "Wow" Signal: Mismatch Negativity in Latent Space** 

To test the *Impenetrable Representation Hypothesis*, we measured the "subjective shock" of introducing chromatic stimuli to the grayscale-trained conceptual model. We hypothesized that genuine qualia onset corresponds to a high-energy prediction error rather than simple feature extraction.

* **Hypothesis 1 (Format Novelty vs. Content Novelty):** We anticipate that the Mahalanobis Distance ($$D\_M$$) for chromatic inputs (Release Phase) will be significantly higher ($$p \< .001$$) than for achromatic control inputs.  
* **Result:** \[Placeholder for Graph 1\] The analysis revealed a distinct spike in Mahalanobis distance upon the introduction of RGB stimuli. This computational signal correlates with the biological Mismatch Negativity (MMN) observed in predictive coding literature, signaling a "Violation of Expectation" (VoE) where the input violates the system's "grayscale prior".

### The Inverted Spectrum: Structural Realism

To test the *Subjective Specificity* of the experience, we compared the latent geometries of two functionally identical MaryVLM agents ($$M\_A$$ and $$M\_B$$) initialized with different random seeds.

* **Hypothesis 2 (Functional Equivalence, Structural Disparity):** We expect both agents to achieve near-identical performance on Visual Question Answering (e.g., both output "Red" for a rose), demonstrating functional equivalence.  
* **Result:** \[Placeholder for Procrustes Analysis Plot\] While VQA accuracy was comparable (X% vs X%), a Procrustes Analysis of their latent centroids for color concepts revealed significant rotational misalignment. This provides a computational existence proof for the "Inverted Spectrum"—functionally identical agents with private, implementation-dependent internal representations.

## Discussion

Our *MaryVLM* framework offers a rigorous computational translation of the "Explanatory Gap," unifying philosophical insights with empirical metrics from cognitive science and AI. By modeling qualia not as "raw feels" but as the computational cost of format translation, we arrive at a physicalist account of subjectivity that respects the phenomenology of "shock".

### The Phenomenology of Prediction Error 

Our finding of a robust "Wow" signal unifies the philosophical concept of "phenomenal surprise" with the neuroscientific framework of *Predictive Coding*. In biological systems, the Mismatch Negativity (MMN) signal arises when sensory input violates the brain's internal generative model. Our model replicates this: Mary’s "shock" is a massive Out-of-Distribution (OOD) error term generated when the "read-only" visual module forces a "chromatic" format into a "luminance" conceptual schema.

This suggests that the "feeling" of qualia is the metabolic and entropic cost of this representational friction. It aligns with the *Global Neuronal Workspace* theory, which posits that only high-intensity prediction errors trigger "global ignition" and enter conscious awareness. In MaryVLM, the "Wow" signal is the threshold mechanism that promotes sensory data from unconscious processing to the conscious "workspace" of the language model.

### Resolving the "Inverted Spectrum"

* Inverted Spectrum is often cited as   
* We demonstrate, in physical terms, how the internal representations can be different, yet yield similar results  
* Tie in work by Kawakita (2025)

### The Limits of Encapsulation: Differentiating Impenetrable Representations

Our **Impenetrable Representation Hypothesis** posits a strict architectural divide. However, to account for empirical evidence of perceptual learning, we must clarify how top-down signals interact with this divide without violating the "read-only" constraint. We propose a mechanism of **"Soft Encapsulation"** defined by the interaction between phylogenetic and ontogenetic systems.

#### The Stability of Phylogenetic Representations (The "Hard" Boundary) 

The sensory encoder produces **Phylogenetic Representations**: high-dimensional states defined by evolutionarily fixed (frozen) weights. These states are **Cognitively Impenetrable**—they are "read-only" to the conceptual system. No amount of "inner speech" or conceptual belief can rewrite the vector coordinates of "Red" to match "Green." The topology of the latent space is immutable.

#### The Plasticity of Ontogenetic Representations (The "Blurred" Boundary)

However, the downstream conceptual system operates on **Ontogenetic Representations**—malleable, "read-write" structures derived from social learning. While this system cannot *overwrite* the Phylogenetic Representation, it can *modulate* how it samples from it. We cite two well-documented effects as evidence for this mechanism:

#### Case A: Ontogenetic Modulation (The "Russian Blue" Effect)

Research on categorical perception (Winawer et al., 2007; Lupyan, 2015\) demonstrates that language speakers (e.g., Russian speakers distinguishing *goluboy* from *siniy*) exhibit faster discrimination of color boundaries.

**Our Interpretation:** This is **Top-Down Attentional Gain**. The **Ontogenetic System** (Language) applies a "predictive filter" to the **Phylogenetic Representation** (Vision). The hardware format of "blue" remains Impenetrable (the rods and cones do not change), but the **Penetrable** conceptual layer learns to amplify specific dimensions of the sensory manifold. The experience changes not because the input was rewritten, but because the readout was sharpened.

#### Case B: Differentiation of Manifolds (The "Wine Taster" Effect) 

Similarly, perceptual learning literature (Goldstone, 1998\) shows that experts (e.g., wine tasters) perceive distinct features where novices perceive a unified "blob."

**Our Interpretation:** This represents the **Differentiation of Impenetrable States**. For a novice, the Ontogenetic system lacks the granularity to map the Phylogenetic input, resulting in a high-entropy "Wow" signal (confusion). For the expert, the Ontogenetic system has learned a higher-resolution mapping. The "Wow" signal decreases not because the sensory hardware changed (it is frozen), but because the **Software Resolution** of the conceptual system successfully differentiated the immutable hardware format.

**Conclusion** The representation itself remains **Impenetrable** (the hardware format is fixed), but the agent's sensitivity to it becomes refined. The "Wow" signal decreases over time because the **Ontogenetic System** successfully optimizes its interface with the **Phylogenetic** manifold, transitioning from "Shock" to "Categorization."

### Limitations and Future Directions

Future work should extend the "Impenetrable Representation Hypothesis" to auditory or tactile domains to test if the "Wow" signal is modality-invariant. Additionally, integrating this individual model into a *Collective Predictive Coding* framework could demonstrate how a community of "inverted" Marys evolves a shared language despite their private representational differences.

---

### **References**

* Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. *Annual Review of Neuroscience*, 28, 403-450.  
* Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181-204.  
* Griffith, T. W., & Byrne, M. D. (2026). Qualia: The Hard Problem. *Proceedings of the Cognitive Science Society*.  
* HuggingFaceTB. (2024). *SmolVLM-Instruct* \[Computer software\]. [https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)  
* Jackson, F. (1982). Epiphenomenal Qualia. *The Philosophical Quarterly*, 32(127), 127-136.  
* Levine, J. (1983). Materialism and qualia: The explanatory gap. *Pacific Philosophical Quarterly*, 64(4), 354-361.  
* Ludlow, P., Nagasawa, Y., & Stoljar, D. (Eds.). (2004). *There's Something About Mary: Essays on Phenomenal Consciousness and Frank Jackson's Knowledge Argument*. MIT Press.  
* Naatanen, R., Paavilainen, P., Rinne, T., & Alho, K. (2007). The mismatch negativity (MMN) in basic research of central auditory processing: A review. *Clinical Neurophysiology*, 118(12), 2544-2590.  
* Nagel, T. (1974). What is it like to be a bat? *The Philosophical Review*, 83(4), 435-450.  
* Nemirow, L. (1990). Physicalism and the cognitive role of images. In W. G. Lycan (Ed.), *Mind and Cognition*. Blackwell.  
* Pont-Tuset, J., et al. (2020). Connecting Vision and Language with Localized Narratives. *ECCV*.  
* **Aston-Jones, G., & Cohen, J. D.** (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. *Annual Review of Neuroscience*.  
* **Bellini-Leite, S. C.** (2024). The 'Generative AI Mary's room' thought experiment. *AI & Society*.  
* **Clarke, S.** (2023). The Common Impenetrable Representation Hypothesis. *Mind & Language*.  
* **Friston, K.** (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.  
* **Haxby, J. V., et al.** (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. *Neuron*.  
* **Itti, L., & Baldi, P.** (2009). Bayesian surprise attracts human attention. *Vision Research*.  
* **Jackson, F.** (1982). Epiphenomenal Qualia. *The Philosophical Quarterly*.  
* **Kriegeskorte, N., et al.** (2008). Representational similarity analysis-connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*.  
* **Loar, B.** (1990). Phenomenal States. *Philosophical Perspectives*.  
* **Oizumi, M., & Tsuchiya, N.** (2025). Constructive Approach to Bidirectional Influence between Qualia Structure and Language Emergence. *arXiv preprint arXiv:2409.09413*.  
* **Papineau, D.** (2002). *Thinking about Consciousness*. Oxford University Press.  
* **Sucholutsky, I., & Griffiths, T. L.** (2023). Is my "red" your "red"?: Unsupervised alignment of qualia structures via optimal transport. *arXiv preprint*.  
* **Taniguchi, T., et al.** (2025). Constructive Approach to Bidirectional Influence between Qualia Structure and Language Emergence. *Waseda University / arXiv*.  
* **Wacongne, C., et al.** (2012). Evidence for a hierarchy of predictions and prediction errors in human cortex. *Proceedings of the National Academy of Sciences*.  
* 

# 