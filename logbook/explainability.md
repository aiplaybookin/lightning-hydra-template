<table align="center"><tr><td align="center" width="9999">

# Model Explainability

Survey of different methods of model explainability in Computer Vision

</td></tr></table>

<table align="center"><tr><td align="left" width="9999">

# To generate explainability results

1. Generate images from [DreamStudio by Stability AI (using stable diffusion model)](https://beta.dreamstudio.ai/)
2. Place images in ```logbook/images/``` folder
3. Run ```explain.py``` with model path or use timm models ( can also refer notebook placed in ```notebook``` folder)
   - Resulting images will be generated and stored in ```logbook/outputs/```
4. Run ```robustness.py```
   - Resulting images will be generated and stored in ```logbook/outputs/```

</td></tr></table>

<table align="center"><tr><td align="center" width="9999">

# Results 

## Integrated Gradients
<img src="outputs/1_IG1.png" align="center" width="450" >
<img src="outputs/1_IG2.png" align="center" width="450" >
<img src="outputs/1_IG3.png" align="center" width="450" >

## Integrated Gradients with Noise Tunnel
<img src="outputs/2_IGN1.png" align="center" width="450" >
<img src="outputs/2_IGN2.png" align="center" width="450" >
<img src="outputs/2_IGN3.png" align="center" width="450" >

## SHAP
<img src="outputs/3_SHAP1.png" align="center" width="450" >
<img src="outputs/3_SHAP2.png" align="center" width="450" >
<img src="outputs/3_SHAP3.png" align="center" width="450" >


## Occlusion
<img src="outputs/4_OCC1.png" align="center" width="450" >
<img src="outputs/4_OCC2.png" align="center" width="450" >
<img src="outputs/4_OCC3.png" align="center" width="450" >

## Grad CAM & CAM++

Note : Grad CAM++ works well with multiple items, see mushrooms

<img src="outputs/5_GRADCAM1.png" align="center" width="450" >
<img src="outputs/5_GRADCAM2.png" align="center" width="450" >
<img src="outputs/5_GRADCAM3.png" align="center" width="450" >


</td></tr></table>

