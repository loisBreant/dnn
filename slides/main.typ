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

== The Challenge & Innovation

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 2em,
  [
    *Existing Challenges*
    - Low-light images suffer from low visibility, noise, and color cast.
    - *Supervised Methods:* Require expensive paired data; often overfit to specific sensors.
    - *Unsupervised (GANs):* Hard to train; require carefully selected unpaired datasets.
    
    #v(1em)
    
    *The Zero-DCE Breakthrough*
    - *Zero-Reference:* No paired or unpaired data needed.
    - *Reformulation:* Enhancement as an image-specific curve estimation task.
  ],
  [
    *Key Contributions*
    - *LE-curve:* A pixel-wise, high-order curve for dynamic adjustment.
    - *DCE-Net:* A lightweight CNN (~79k parameters) that estimates curve parameters.
    - *Non-reference Losses:* A suite of functions to train without ground truth.

    #v(1em)
    #align(center)[
      #box(fill: orange.lighten(80%), inset: 1em, radius: 8pt)[
        *Core Principle:*\
        Pixel mapping via $n^{t h}$-order differentiable curves
      ]
    ]
  ]
)

= Methodology

== Light-Enhancement Curve (LE-curve)

The core mechanism is a quadratic curve designed for light enhancement:
$ L E(I(x); alpha) = I(x) + alpha I(x)(1 - I(x)) $

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Design Constraints:*
    - *Self-regularized:* $L E(0)=0$ and $L E(1)=1$.
    - *Monotonicity:* Preserves contrast in local regions.
    - *Differentiable:* Enables end-to-end training via backpropagation.
  ],
  [
    *Higher-Order Iteration ($n=8$):*
    $ I_n (x) = L E(I_{n-1}(x); cal(A)_n (x)) $
    Where $cal(A)_n$ is the pixel-wise parameter map estimated by the network.
  ]
)



== DCE-Net & Training Strategy

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Architecture:*
    - 7 convolutional layers.
    - Symmetrical skip-connections (concatenation).
    - Output: 24 parameter maps (8 iterations $times$ 3 RGB channels).
    
    #v(1em)
    *Data:*
    - Trained on the SICE dataset (part 1).
    - Only uses low-light images for training—*no ground truth used.*
  ],
  [
    *Total Loss:* $L_{t o t a l} = L_{s p a} + L_{e x p} + W_{c o l} L_{c o l} + W_{t v_cal(A)} L_{t v_cal(A)}$
    
    #v(0.5em)
    *Non-Reference Loss Functions:*
    1. *Spatial Consistency ($L_{s p a}$):* Preserves neighbor differences.
    2. *Exposure Control ($L_{e x p}$):* Drives pixels toward a target level (0.6).
    3. *Color Constancy ($L_{c o l}$):* Prevents color shifts.
    4. *Illumination Smoothness ($L_{t v}$):* Ensures smooth parameter transitions.
  ]
)

= Results & Performance

== Quantitative Comparison

#align(center)[
  #table(
    columns: 5,
    align: center,
    [*Method*], [*Params*], [*FLOPs*], [*PSNR↑*], [*SSIM↑*],
    [LIME], [-], [-], [16.76], [0.56],
    [RetinexNet], [0.84M], [587.47G], [16.77], [0.56],
    [EnlightenGAN], [18.94M], [170.64G], [17.48], [0.65],
    [*Zero-DCE*], [*0.079M*], [*5.21G*], [*14.86*], [*0.56*],
  )
]

#v(1em)

*Performance Insights:*
- *Inference Speed:* ~500 FPS on a GTX 1080Ti (Real-time).
- *Efficiency:* 100x fewer parameters than EnlightenGAN.
- *Robustness:* Better visual quality in extreme dark/backlit scenes (User Study leader).

== Visual Results

#figure(
  image("images/fig7.png",width: 80%, height: 80%),
  caption: [Zero-DCE achieves natural brightness without over-saturation or artifacts.],
)

= Conclusion

== Summary & Impact

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Strengths*
    - *Efficiency:* Smallest model in its class.
    - *Generality:* Works across various lighting and sensor types.
    - *Novelty:* Proves that deep learning can succeed without labels or GANs.
    
    #v(0.5em)
    
    *Applications*
    - Mobile photography enhancement.
    - Pre-processing for object detection.
    - Video surveillance in the dark.
  ],
  [
    *Limitations*
    - Does not explicitly perform denoising.
    - Fixed iteration number ($n=8$) may not suit all scenes.
    
    #v(0.5em)
    
    *Future Directions*
    - Integrating noise reduction into the curve.
    - Adaptive iteration based on image content.
    - Applying Zero-DCE to other tasks like dehazing.
  ]
)

#v(1em)

#align(center)[
  #text(size: 1.5em, weight: "bold")[Questions?]
]

#v(0.5em)

#align(center)[
  #text(size: 0.8em, style: "italic")[
    Guo et al. (2020) CVPR, 10.1109/CVPR42600.2020.00944
  ]
]