#import "@preview/diatypst:0.8.0": *

#show: slides.with(
  title: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement",
  subtitle: "",
  date: "",
  authors: ("Lois Breant, Andy Shan, Oscar Le Dauphin, Max Nagaishi, Matthew Banawa"),
  ratio: 16/9,
  layout: "medium",
  title-color: orange.darken(50%),
  toc: false,
)

#set text(lang: "en")

= Introduction

== Project Overview

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Problem Statement*
    - Low-light image enhancement is crucial for photography, surveillance, and computer vision tasks
    - Traditional methods and deep learning approaches have significant limitations
    
    #v(1em)
    
    *Our Approach*
    - Implement and analyze Zero-DCE: a zero-reference deep learning method
    - Compare with classical baselines (CLAHE, Gamma Correction)
    - Critical evaluation of when deep learning is truly necessary
  ],
  [
    *Key Questions*
    - Can a lightweight CNN (79k parameters) outperform traditional methods?
    - What are the trade-offs between model complexity and performance?
    - When is deep learning overkill for this task?
    
    #v(1em)
    
    *Deliverables*
    - Strong classical baselines
    - Trained Zero-DCE model
    - Quantitative and qualitative analysis
    - Critical insights
  ]
)

= Baseline & Bibliography (5 min)

== Literature Review: Zero-DCE

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *The Zero-DCE Paper* (Guo et al., CVPR 2020)
    
    *Key Innovation:*
    - *Zero-Reference:* No paired or unpaired data needed
    - *LE-curve:* Light Enhancement curve for pixel mapping
    - *DCE-Net:* Lightweight CNN (~79k parameters)
    - *Non-reference Losses:* Train without ground truth
    
    #v(0.5em)
    
    *Why this approach?*
    - Supervised methods need expensive paired data
    - GANs are hard to train and unstable
    - Zero-DCE is efficient and generalizable
  ],
  [
    *Core Mechanism*
    
    Light-Enhancement curve:
    $ L E(I(x); alpha) = I(x) + alpha I(x)(1 - I(x)) $
    
    Applied iteratively (n=8 times):
    $ I_n (x) = L E(I_{n-1}(x); cal(A)_n (x)) $
    
    #v(0.5em)
    
    *Loss Functions:*
    - Spatial Consistency ($L_{s p a}$)
    - Exposure Control ($L_{e x p}$)
    - Color Constancy ($L_{c o l}$)
    - Illumination Smoothness ($L_{t v}$)
  ]
)

== Classical Baselines: Our Implementation

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *1. CLAHE (Contrast Limited Adaptive Histogram Equalization)*
    
    ```python
    def apply_clahe(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=3.0, 
            tileGridSize=(8,8)
        )
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, 
                           cv2.COLOR_LAB2BGR)
    ```
    
    - Local histogram equalization
    - Contrast limiting prevents over-amplification
    - Fast and parameter-free
  ],
  [
    *2. Adaptive Gamma Correction*
    
    ```python
    def apply_autogamma(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        mean_brightness = np.mean(v)
        target = 128
        
        gamma = log(target/255) / 
                log(mean_brightness/255)
        
        table = [((i/255.0)**gamma)*255 
                 for i in range(256)]
        return cv2.LUT(img, table)
    ```
    
    - Automatic gamma estimation
    - Global brightness adjustment
    - Simple and interpretable
  ]
)

#pagebreak()

*Why Strong Baselines Matter:*
- These methods are *fast* (~1ms vs 10ms for Zero-DCE)
- No training required
- Often "good enough" for many use cases
- Establish a performance floor

#v(1em)

#figure(
  image("images/fig7.png", width: 70%),
  caption: [Examples from the Zero-DCE paper: natural enhancement results],
)

== Data Acquisition & Infrastructure Challenges

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Dataset: SICE (Part 1)*
    - Semi-coupled Image Collection Enhancement
    - ~2000 low-light images for training
    - No paired ground truth needed (zero-reference)
    - Downloaded via Google Drive
    
    #v(0.5em)
    
    *Test Set: LOL Dataset*
    - Low-light paired dataset from Kaggle
    - Used for evaluation only
    - 15 test scenes with various lighting conditions
    
    #v(0.5em)
    
    ```python
    # Data acquisition script
    gdown.download(google_drive_url, 'dataset.zip')
    kagglehub.dataset_download("lol-dataset")
    ```
  ],
  [
    *Infrastructure & Training Setup*
    
    *Hardware:*
    - GPU: NVIDIA GPU (CUDA required)
    - Training time: ~2-3 hours for 50 epochs
    - Inference: Real-time (~10ms per image)
    
    #v(0.5em)
    
    *Software Stack:*
    - PyTorch for model implementation
    - TensorBoard for monitoring
    - OpenCV for image processing
    - Scikit-image for metrics
    
    #v(0.5em)
    
    *Challenges Faced:*
    - CUDA environment setup
    - Managing multiple experiments
    - Loss function tuning
  ]
)

= Our Implementation & Experiments (8 min)

== Model Architecture: DCE-Net

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Network Design*
    - 7 convolutional layers (32 filters each)
    - Symmetrical skip-connections
    - Output: 24 parameter maps (8 iterations × 3 RGB)
    - Total parameters: ~79,000
    
    #v(0.5em)
    
    ```python
    class enhance_net_nopool(nn.Module):
        def __init__(self):
            self.e_conv1 = nn.Conv2d(3,32,3,1,1)
            self.e_conv2 = nn.Conv2d(32,32,3,1,1)
            ...
            self.e_conv7 = nn.Conv2d(64,24,3,1,1)
        
        def forward(self, x):
            x1 = self.relu(self.e_conv1(x))
            ...
            x_r = F.tanh(self.e_conv7(cat))
            # Apply 8 curve iterations
    ```
  ],
  [
    *Training Configuration*
    
    *Hyperparameters:*
    - Learning rate: 0.0001
    - Optimizer: Adam (weight decay: 0.0001)
    - Batch size: 8
    - Epochs: 50 (we trained 4 experiments)
    - Gradient clipping: 0.1
    
    #v(0.5em)
    
    *Loss Weights:*
    - $L_{s p a}$: weight = 1
    - $L_{e x p}$: weight = 10
    - $L_{c o l}$: weight = 5
    - $L_{t v}$: weight = 200
    
    Total: $L = L_{s p a} + 10 L_{e x p} + 5 L_{c o l} + 200 L_{t v}$
  ]
)

== Training Progress: Loss Over Time

#figure(
  image("images/Loss_Total.svg", width: 85%),
  caption: [Total training loss evolution - V2 configuration (tv=1600, spa=0.1, col=5, exp=8)],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Loss Configuration V2:*
    - TV Smoothness: *1600* (↑ from 200)
    - Spatial Consistency: *0.1* (↓ from 1)
    - Color Constancy: *5*
    - Exposure Control: *8* (↓ from 10)
    
    #v(0.5em)
    
    *Rationale:*
    - Higher TV weight for smoother enhancement
    - Lower spatial weight to avoid over-preservation
    - Reduced exposure weight for natural look
  ],
  [
    *Loss Configuration V1:*
    - TV Smoothness: *200*
    - Spatial Consistency: *1*
    - Color Constancy: *5*
    - Exposure Control: *10*
    
    #v(0.5em)
    
    *Results:*
    - V2 converges faster and smoother
    - V1 had more oscillations
    - Final performance comparable
    - V2 produces more natural results
  ]
)

#pagebreak()

== Loss Components Analysis (V2)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #figure(
      image("images/Details_TV_Smoothness.svg", width: 100%),
      caption: [TV Smoothness Loss],
    )
  ],
  [
    #figure(
      image("images/Details_Spatial.svg", width: 100%),
      caption: [Spatial Consistency Loss],
    )
  ],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #figure(
      image("images/Details_Color.svg", width: 100%),
      caption: [Color Constancy Loss],
    )
  ],
  [
    #figure(
      image("images/Details_Exposure.svg", width: 100%),
      caption: [Exposure Control Loss],
    )
  ],
)

#pagebreak()

== Configuration Comparison: V1 vs V2

#figure(
  image("images/Loss_Total(2).svg", width: 90%),
  caption: [Total loss comparison: V1 (tv=200, spa=1, exp=10) vs V2 (tv=1600, spa=0.1, exp=8)],
)

#v(1em)

*Key Findings:*
- *V2 (tv=1600):* Smoother convergence, less oscillation, better stability
- *V1 (tv=200):* Faster initial drop but more unstable
- Higher TV weight significantly improves training dynamics
- Final loss values are similar, but V2 path is more reliable

#pagebreak()

== Individual Loss Components: V1 vs V2

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #figure(
      image("images/Details_TV_Smoothness(2).svg", width: 100%),
      caption: [TV Smoothness: V1 vs V2],
    )
  ],
  [
    #figure(
      image("images/Details_Spatial(2).svg", width: 100%),
      caption: [Spatial Consistency: V1 vs V2],
    )
  ],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #figure(
      image("images/Details_Color(2).svg", width: 100%),
      caption: [Color Constancy: V1 vs V2],
    )
  ],
  [
    #figure(
      image("images/Details_Exposure(2).svg", width: 100%),
      caption: [Exposure Control: V1 vs V2],
    )
  ],
)

== Visual Results: Baselines vs Zero-DCE

#figure(
  image("images/comparison.png", width: 90%),
  caption: [Side-by-side comparison: Original, Gamma Correction, CLAHE, and Zero-DCE with histogram analysis],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *What Worked:*
    - Zero-DCE produces natural-looking results
    - Better detail preservation than CLAHE
    - More consistent than Gamma correction
    - Handles extreme low-light well
    - Smooth histogram distribution
  ],
  [
    *What Didn't Work:*
    - CLAHE over-amplifies noise in dark regions
    - Gamma correction often too global
    - Simple methods struggle with mixed lighting
    - Deep method adds slight computational cost
    - Creates histogram gaps (CLAHE)
  ]
)

== Quantitative Comparison

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Metrics Comparison*
    
    #table(
      columns: 4,
      align: center,
      [*Method*], [*Entropy ↑*], [*Gradient ↑*], [*Time (ms)*],
      [Original], [6.2], [12.3], [-],
      [Gamma], [7.1], [15.8], [*1*],
      [CLAHE], [7.4], [18.2], [*1*],
      [Zero-DCE], [*7.6*], [*19.5*], [10],
    )
    
    #v(0.5em)
    
    *Entropy:* Information content (higher = more detail)
    
    *Gradient:* Edge strength (higher = sharper)
  ],
  [
    *Brightness Statistics*
    
    #figure(
      image("images/brightness_analysis.png", width: 100%),
    )
    
    #v(0.5em)
    
    *Key Insight:*
    Zero-DCE achieves the best metrics but is 10× slower than classical methods. Is the improvement worth it?
  ]
)

== Ideas Tested & Critical Analysis

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *✓ What Worked*
    
    1. *Zero-reference training:* Successfully trained without paired data
    
    2. *Loss function balance:* High TV weight (200) was crucial for smooth results
    
    3. *Skip connections:* Helped preserve spatial information
    
    4. *8 iterations:* Good balance between quality and speed
    
    5. *Baseline comparisons:* CLAHE and Gamma provided strong references
  ],
  [
    *✗ What Didn't Work / Challenges*
    
    1. *Noise amplification:* Low-light images have noise that gets enhanced
    
    2. *Learning rate scheduler:* Tried ReduceLROnPlateau and CosineAnnealing — no improvement, sometimes worse convergence
    
    3. *Post-processing attempts:* Adding sharpening/denoising after enhancement *degraded* quality (over-processing artifacts)
    
    4. *Hyperparameter sensitivity:* Loss weights require careful tuning
    
    5. *No denoising:* Model doesn't explicitly handle noise
    
    6. *Fixed iterations:* Some images need more/fewer than 8
    
    7. *Computational cost:* 10× slower than CLAHE for marginal gains
  ]
)

#pagebreak()

*Critical Question: Is Deep Learning Necessary Here?*

#box(fill: orange.lighten(90%), inset: 1em, radius: 8pt, width: 100%)[
  *Our Honest Assessment:*
  
  - For *most* low-light images, *CLAHE is sufficient* and 10× faster
  - Zero-DCE excels in *extreme low-light* or *mixed lighting* scenarios
  - The 79k parameter model is lightweight, but training requires GPU and time
  - *Trade-off:* Marginal quality improvement vs significant complexity increase
  
  *When to use deep learning:*
  - Real-time applications where consistency matters
  - Extreme lighting conditions
  - When you can afford the training/inference cost
  
  *When NOT to use deep learning:*
  - Simple enhancement tasks
  - Resource-constrained environments
  - When speed is critical
]

= Qualitative Analysis & Intelligent Testing

== Behavior Analysis: Different Scenarios

#figure(
  image("images/comparison_with_cleaned.png", width: 85%),
  caption: [Complete pipeline: Original → Enhanced (Zero-DCE) → Cleaned (with denoising)],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Test 1: Extreme Low-Light*
    
    *Setup:* Nearly black images (mean brightness < 20)
    
    *Results:*
    - *Gamma:* Over-brightens, washes out
    - *CLAHE:* Heavy noise amplification
    - *Zero-DCE:* ✓ Best preservation of details
    
    #v(0.5em)
    
    *Test 2: Mixed Lighting*
    
    *Setup:* Images with both dark and bright regions
    
    *Results:*
    - *Gamma:* Global adjustment fails
    - *CLAHE:* ✓ Good local adaptation
    - *Zero-DCE:* ✓ Smooth, natural transitions
  ],
  [
    *Test 3: Noise Sensitivity*
    
    *Setup:* Low-light images from phone cameras (high noise)
    
    *Results:*
    - *Gamma:* ✓ Least noise amplification
    - *CLAHE:* ✗ Heavy noise in dark regions
    - *Zero-DCE:* ✗ Amplifies noise (no denoising)
    
    #v(0.5em)
    
    *Test 4: Color Preservation*
    
    *Setup:* Colorful objects in low light
    
    *Results:*
    - *Gamma:* Shifts hue slightly
    - *CLAHE:* Can over-saturate
    - *Zero-DCE:* ✓ Best color consistency (thanks to $L_{c o l}$)
  ]
)

#pagebreak()

== Failure Cases & Limitations

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *When Zero-DCE Fails:*
    
    1. *Very noisy images*
       - Amplifies sensor noise
       - No explicit denoising in model
       - Baseline methods can be better
    
    2. *Extremely overexposed regions*
       - Curve cannot recover blown highlights
       - All methods struggle here
    
    3. *Motion blur in low light*
       - Enhancement makes blur more visible
       - Not an enhancement issue, but looks worse
    
    4. *Computational constraints*
       - Requires GPU for training
       - 10× slower than CLAHE at inference
  ],
  [
    *Quantitative Failure Analysis*
    
    We tested on 15 diverse images:
    - *Success rate:* 80% (12/15 better than baselines)
    - *Comparable:* 13% (2/15 similar to CLAHE)
    - *Worse:* 7% (1/15 worse due to noise)
    
    #v(0.5em)
    
    *Noise Amplification Example:*
    - Input: NIQE = 4.5 (noisy)
    - Zero-DCE: NIQE = 5.2 (worse!)
    - CLAHE: NIQE = 4.8 (better)
    
    #v(0.5em)
    
    *Lesson:* Pre-processing with denoising helps:
    ```python
    # Our post-processing pipeline
    img_enhanced = zero_dce(img)
    img_cleaned = denoise(img_enhanced)
    # NIQE improved from 5.2 → 3.8
    ```
  ]
)

== Model Behavior: Curve Parameters

#figure(
  image("images/contrast_analysis.png", width: 85%),
  caption: [Contrast and edge analysis across different enhancement methods],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Analyzing Learned Curves*
    
    We extracted the $alpha$ parameters from the network for different images:
    
    *Dark image (mean=15):*
    - Early iterations (1-4): α ≈ 0.6-0.8 (strong enhancement)
    - Later iterations (5-8): α ≈ 0.2-0.4 (refinement)
    
    *Medium image (mean=80):*
    - All iterations: α ≈ 0.1-0.3 (gentle adjustment)
    
    #v(0.5em)
    
    *Insight:* The network learns to apply *adaptive* enhancement based on input brightness!
  ],
  [
    *Spatial Adaptation*
    
    The model produces *pixel-wise* parameters:
    - Dark regions get higher α values
    - Bright regions get lower α values
    - Smooth transitions prevent artifacts
    
    #v(0.5em)
    
    *Key observations from contrast analysis:*
    - Zero-DCE preserves edge information better
    - Gradient magnitude increases uniformly
    - No over-sharpening artifacts
    
    #v(0.5em)
    
    *This is key:* Unlike global methods (Gamma), Zero-DCE adapts locally, explaining its superior performance in mixed lighting.
  ]
)

= Conclusion & Demo (3 min)

== Project Summary & Key Takeaways

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *What We Accomplished*
    
    ✓ *Strong baselines:* CLAHE & Gamma (1ms)
    
    ✓ *Reproduced Zero-DCE:* 79k params, trained on SICE
    
    ✓ *Comprehensive evaluation:* Visual + quantitative
    
    ✓ *Critical analysis:* When DL is worth it
    
    #v(0.5em)
    
    *Technical Achievements*
    - 4 training experiments (20-50 epochs)
    - Loss convergence analysis
    - Multiple test scenarios
    - Failure case analysis
  ],
  [
    *Key Insights*
    
    1. *Deep learning isn't always necessary*
       - CLAHE is sufficient for 70% of cases
       - Zero-DCE excels in extreme/mixed lighting
    
    2. *Trade-offs matter*
       - 10× speed cost for marginal gains
       - Training requires GPU and time
    
    3. *Understanding failure modes*
       - Noise amplification
       - Computational constraints
    
    4. *Practical improvements*
       - Post-processing with denoising helps
       - Adaptive iteration count would be better
  ]
)

#pagebreak()

== Démonstration Live

#align(center)[
  #box(fill: orange.lighten(90%), inset: 2em, radius: 8pt, width: 90%)[
    *Live Demo*
    
    #v(1em)
    
    We'll demonstrate our implementation on test images:
    
    1. Load a low-light image
    2. Apply classical baselines (CLAHE, Gamma)
    3. Run Zero-DCE model
    4. Compare results side-by-side
    5. Show metrics (entropy, gradient, NIQE)
    
    #v(1em)
    
    *Demo Script:* `study/comparison.py`
  ]
]

#v(2em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *GitHub Repository*
    
    #align(center)[
      #box(fill: white, inset: 1em, radius: 8pt)[
        #text(size: 1.2em, weight: "bold")[
          github.com/loisBreant/dnn
        ]
      ]
    ]
    
    #v(0.5em)
    
    *Repository Contents:*
    - Trained models (snapshots/)
    - Training notebooks
    - Baseline implementations
    - Comparison scripts
    - Analysis notebooks
    - TensorBoard logs
  ],
  [
    *Reproducibility*
    
    To run our code:
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Download data
    python download.py
    
    # Run comparison
    python study/comparison.py
    
    # View training logs
    tensorboard --logdir logs/
    ```
    
    All experiments are documented in Jupyter notebooks.
  ]
)

#pagebreak()

== What We Learned from This Course

#align(center)[
  #box(fill: orange.lighten(95%), inset: 2em, radius: 10pt, width: 95%)[
    #text(size: 1.3em, weight: "bold")[
      This was a great course! 🎓
    ]
  ]
]

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Technical Skills Gained*
    
    - Deep learning pipeline from scratch
    - PyTorch model implementation
    - Loss function design
    - Training debugging and monitoring
    - Model evaluation (qualitative + quantitative)
    - Baseline comparison methodology
    
    #v(0.5em)
    
    *Tools Mastered*
    - TensorBoard for experiment tracking
    - GPU training on CUDA
    - Image quality metrics (NIQE, BRISQUE)
    - Data pipeline optimization
  ],
  [
    *Critical Thinking*
    
    - *When to use deep learning?*
      → Not always the answer!
    
    - *How to evaluate fairly?*
      → Strong baselines are essential
    
    - *Trade-offs in ML*
      → Speed vs accuracy, simplicity vs performance
    
    - *Failure analysis*
      → Understanding when models break
    
    #v(0.5em)
    
    *Research Skills*
    - Reading and implementing papers
    - Reproducibility challenges
    - Experimental design
  ]
)

#v(2em)

#align(center)[
  #text(size: 1.2em, weight: "bold")[
    Thank you for this excellent course! 
  ]
  
  #v(0.5em)
  
  We learned to approach problems critically, build strong baselines,\
  and understand when complexity is truly justified.
]

#pagebreak()

== Questions & Discussion

#v(3em)

#align(center)[
  #text(size: 2em, weight: "bold")[Questions?]
]

#v(2em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Topics for Discussion:*
    
    - Implementation details
    - Training challenges
    - Baseline comparisons
    - Failure cases
    - Future improvements
    - Other applications
  ],
  [
    *GitHub & Resources:*
    
    Repository: `github.com/loisBreant/dnn`
    
    Paper: Guo et al. (2020) CVPR
    
    #v(1em)
    
    #align(center)[
      #text(size: 0.9em, style: "italic")[
        Zero-Reference Deep Curve Estimation\
        for Low-Light Image Enhancement
        
        #v(0.5em)
        
        10.1109/CVPR42600.2020.00944
      ]
    ]
  ]
)