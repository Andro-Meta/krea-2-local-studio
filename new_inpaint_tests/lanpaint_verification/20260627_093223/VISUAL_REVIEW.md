# LanPaint Inpaint Visual Review

Fixed seed prompt: replace red object with glossy blue ceramic vase.

## Results
- native: completed.
- lanpaint: completed.

## Default Recommendation
- Native Krea remains the safest default.
- LanPaint should be marked experimental but usable for hard-mask inpaint with 20+ diffusion steps, 5 think steps, lambda 16, step size 0.2, beta 1, friction 15, early stop 1.
- Use hard binary masks for LanPaint. Soft masks are converted to binary before sampling.