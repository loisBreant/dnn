#import "@preview/diatypst:0.8.0": *

#show: slides.with(
  title: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement",
  subtitle: "Critical Analysis and Implementation",
  date: "January 2026",
  authors: ("Lois Breant, Andy Shan, Oscar Le Dauphin, Max Nagaishi, Matthew Banawa"),
  ratio: 16/9,
  layout: "medium",
  title-color: rgb("#1a5fb4"), // Professional blue
  toc: false,
)

// Footer with page numbering
#set page(
  footer: context [
    #set text(size: 10pt, fill: gray)
    #counter(page).display("1 / 1", both: true)
    #h(1fr)
    Zero-DCE Project
  ]
)

#set text(lang: "en", size: 20pt)

= Introduction

== Project Overview

#grid(
  columns: (1fr, 1fr),
  gutter: 3em,
  [
    *The Problem*
    - Low-light images suffer from low visibility and high noise.
    - Traditional methods often lack local adaptation.
    - Supervised DL requires expensive paired datasets.
    
    #v(1em)
    
    *Our Objective*
    - Implement *Zero-DCE* (Zero-Reference).
    - Compare with *CLAHE* and *Gamma* baselines.
    - Evaluate if DL is truly justified for this task.
  ],
  [
    *Key Questions*
    - Can a lightweight CNN (79k params) beat classical math?
    - What are the real-world trade-offs?
    - Performance vs. Computational cost.
    
    #v(1em)
    
    *Outcome*
    - Quantitative metrics (Entropy, Gradient).
    - Qualitative analysis of artifacts.
    - Hybrid denoising experiments.
  ]
)

= Baseline & Bibliography

== Literature Review: Zero-DCE

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Zero-DCE Innovation* (CVPR 2020)
    
    - *Zero-Reference:* No ground truth images needed.
    - *Pixel-wise Curve:* Learns an enhancement curve $alpha$ for every pixel.
    - *Iterative:* Curve applied 8 times for gradual enhancement.
    - *Lightweight:* Only ~79k parameters.
  ],
  [
    *Loss Strategy*
    - *Spatial:* Preserves neighbor relations.
    - *Exposure:* Maintains brightness levels.
    - *Color:* Prevents color shifts.
    - *Smoothness:* Ensures natural transitions.
    
    #v(1em)
    #box(fill: gray.lighten(90%), inset: 10pt, radius: 5pt)[
      _Refer to Appendix A for the LE-Curve mathematical formulation._
    ]
  ]
)

== Classical Baselines

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *1. Adaptive Gamma Correction*
    - Global adjustment.
    - Uses mean brightness to estimate $gamma$.
    - Fast, but lacks local detail.
    
    #v(1em)
    *2. CLAHE*
    - Local histogram equalization.
    - Clips contrast to avoid noise explosion.
    - Strong baseline for local contrast.
  ],
  [
    #figure(
      image("images/fig7.png", width: 90%),
      caption: [Reference enhancement (Source: Guo et al.)],
    )
  ]
)

== Data & Infrastructure

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Datasets*
    - *Training:* SICE (2000+ low-light images).
    - *Testing:* LOL Dataset (15 scenes).
    
    #v(1em)
    *Hardware*
    - NVIDIA GPU (CUDA) for training.
    - ~3 hours for convergence (50 epochs).
  ],
  [
    *Software Stack*
    - *PyTorch:* Neural Network logic.
    - *TensorBoard:* Loss monitoring.
    - *OpenCV / Scikit-image:* Metrics and Baselines.
    
    #v(1em)
    #box(fill: gray.lighten(90%), inset: 10pt, radius: 5pt)[
      _Refer to Appendix B for data ingestion logic._
    ]
  ]
)

= Implementation & Experiments

== Model: DCE-Net

- *Architecture:* 7 Convolutional layers with skip-connections.
- *Output:* 24 maps (8 iterations $times$ 3 RGB channels).
- *Efficiency:* Real-time inference (~10ms).

#v(1em)

#align(center)[
  #table(
    columns: (auto, auto),
    stroke: none,
    fill: (x, y) => if y == 0 { gray.lighten(80%) } else { white },
    [*Hyperparameter*], [*Value*],
    [Learning Rate], [0.0001],
    [Batch Size], [8],
    [Epochs], [50],
    [Iterations], [8]
  )
]

== Training: Total Loss (V2)

#figure(
  image("images/Loss_Total.svg", width: 75%),
  caption: [Total Loss Convergence (V2: High Smoothness weight)],
)

#v(0.5em)
*Observation:* Stable convergence achieved with $W_{t v}=1600$.

== Detailed Loss Analysis (V2)

#grid(
  columns: (1fr, 1fr),
  rows: (1fr, 1fr),
  gutter: 10pt,
  figure(image("images/Details_TV_Smoothness.svg", width: 95%), caption: [TV Smoothness]),
  figure(image("images/Details_Spatial.svg", width: 95%), caption: [Spatial Consistency]),
  figure(image("images/Details_Color.svg", width: 95%), caption: [Color Constancy]),
  figure(image("images/Details_Exposure.svg", width: 95%), caption: [Exposure Control]),
)

== V1 vs V2 Comparison

#figure(
  image("images/Loss_Total(2).svg", width: 75%),
  caption: [Comparison: V1 (Standard weights) vs V2 (High Smoothness)],
)

#v(0.5em)
*Insight:* V2 (blue) shows significantly better stability in the later stages of training.

== Visual Results: Comparison

#figure(
  image("images/comparison.png", width: 85%),
  caption: [Side-by-side comparison with histogram analysis],
)

== Quantitative Performance

#align(center)[
  #table(
    columns: (1.5fr, 1fr, 1fr, 1fr),
    align: center,
    [*Method*], [*Entropy ↑*], [*Gradient ↑*], [*Time*],
    [Original], [6.21], [12.34], [-],
    [Gamma], [7.12], [15.82], [< 1ms],
    [CLAHE], [7.45], [18.19], [< 1ms],
    [*Zero-DCE*], [*7.62*], [*19.48*], [10ms],
  )
]

#v(1em)
*Conclusion:* Zero-DCE wins on quality but is 10$times$ slower than classical baselines.

= Critical Analysis

== What Worked & Challenges

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Successes*
    - Zero-reference training is highly effective.
    - Local adaptation handles mixed lighting perfectly.
    - Color constancy loss preserves natural hues.
  ],
  [
    *Remaining Challenges*
    - *Noise:* Model amplifies existing sensor noise.
    - *Over-processing:* Post-sharpening can create artifacts.
    - *Speed:* 10ms is fast, but 1ms (CLAHE) is better for mobile.
  ]
)

== Is Deep Learning Necessary?

#align(center)[
  #box(fill: blue.lighten(95%), inset: 1.5em, radius: 10pt, width: 90%)[
    *Our Conclusion:*
    
    - For 80% of consumer photos, *CLAHE is sufficient*.
    - Zero-DCE is superior for *extreme low-light* and *professional pipelines*.
    - Hybrid approaches (Enhancement + Denoising) are required for real-world use.
  ]
]

= Annexes

== Appendix A: Mathematical Formulation

*Light-Enhancement curve:*
$ L E(I(x); alpha) = I(x) + alpha I(x)(1 - I(x)) $

*Recursive application:*
$ I_n (x) = L E(I_{n-1}(x); cal(A)_n (x)) $

*Loss Weights (V2):*
- Total: $L = 0.1 L_{s p a} + 8 L_{e x p} + 5 L_{c o l} + 1600 L_{t v}$

== Appendix B: Core Implementation (Baselines)

#set text(size: 14pt)
*CLAHE Implementation:*
```python
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
```

*Adaptive Gamma:*
```python
def apply_autogamma(img):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
    gamma = log(128/255) / log(np.mean(v)/255)
    table = [((i/255.0)**gamma)*255 for i in range(256)]
    return cv2.LUT(img, table)
```

== Appendix C: Model Definition

#set text(size: 12pt)
```python
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # ... layers 2 to 6 ...
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # ... feature extraction & concat ...
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        # Apply curves iteratively
        return enhanced_image
```
