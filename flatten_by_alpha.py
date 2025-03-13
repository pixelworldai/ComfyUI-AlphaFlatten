class FlattenByAlpha:
    """
    A node that flattens a batch of images with alpha channels into a single image.
    Similar to Photoshop's flatten function, this combines multiple masked images
    respecting their alpha channels.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "background_color": (["black", "white", "transparent", "custom"],),
                "custom_color_r": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "custom_color_g": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "custom_color_b": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("flattened_image",)
    FUNCTION = "flatten"
    CATEGORY = "image/processing"

    def flatten(self, images, background_color, custom_color_r=0.0, custom_color_g=0.0, custom_color_b=0.0):
        import torch
        
        # Check if we have a batch of images
        if len(images.shape) != 4:
            raise ValueError("Expected a batch of images with shape [batch, height, width, channels]")
        
        # Check if we have alpha channels (RGBA)
        if images.shape[3] < 4:
            raise ValueError("Images must have alpha channels (RGBA format)")
        
        batch_size, height, width, channels = images.shape
        
        if background_color == "transparent":
            # For transparent background, we need to keep the alpha channel
            # Initialize with a fully transparent image
            result_rgb = torch.zeros((height, width, 3), device=images.device)
            result_alpha = torch.zeros((height, width, 1), device=images.device)
            
            # Process each image in the batch
            for i in range(batch_size):
                img = images[i]
                rgb = img[:, :, :3]
                alpha = img[:, :, 3:4]
                
                # Calculate new alpha for the composite
                new_alpha = result_alpha + alpha * (1 - result_alpha)
                
                # Create a safe mask to avoid division by zero
                mask = new_alpha > 0
                
                # Initialize blended RGB with zeros
                blended_rgb = torch.zeros_like(result_rgb)
                
                # Only blend where the mask is True
                if mask.any():
                    # Apply the alpha blending formula only where mask is True
                    blended_rgb[mask.squeeze()] = (
                        result_rgb[mask.squeeze()] * result_alpha[mask.squeeze()] + 
                        rgb[mask.squeeze()] * alpha[mask.squeeze()] * (1 - result_alpha[mask.squeeze()])
                    ) / new_alpha[mask.squeeze()]
                
                # Update result with the blended values
                result_rgb = torch.where(mask.repeat(1, 1, 3), blended_rgb, result_rgb)
                result_alpha = new_alpha
            
            # Combine RGB and alpha into final result
            result = torch.cat([result_rgb, result_alpha], dim=2)
            
            # Reshape to match expected output format [1, height, width, 4]
            result = result.unsqueeze(0)
            
            return (result,)
        else:
            # For non-transparent backgrounds
            if background_color == "black":
                bg = torch.zeros((height, width, 3), device=images.device)
            elif background_color == "white":
                bg = torch.ones((height, width, 3), device=images.device)
            else:  # custom color
                bg = torch.tensor([custom_color_r, custom_color_g, custom_color_b], device=images.device).view(1, 1, 3).expand(height, width, 3)
            
            # Start with the background
            result = torch.zeros((1, height, width, 3), device=images.device)
            result[0] = bg
            
            # Process each image in the batch
            for i in range(batch_size):
                img = images[i]
                rgb = img[:, :, :3]
                alpha = img[:, :, 3:4]
                
                # Blend the current image with the result using alpha
                result[0] = result[0] * (1 - alpha) + rgb * alpha
            
            return (result,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "FlattenByAlpha": FlattenByAlpha
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlattenByAlpha": "Flatten Images By Alpha"
} 