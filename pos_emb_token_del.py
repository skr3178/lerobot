# Input: 7×7 feature map
image_features = (7, 7, 512)  # 49 locations, 512 features each

# For each location (i,j):
for i in range(7):
    for j in range(7):
        # Get image features for this location
        img_feat = image_features[i, j, :]  # (512,)
        
        # Get positional embedding for this location
        pos_emb = positional_embeddings[i, j, :]  # (512,)
        
        # Combine them
        token = img_feat + pos_emb  # (512,)
        
        # Add to transformer sequence
        transformer_tokens.append(token)

# Result: 49 tokens, each with 512 dimensions
# Each token knows both WHAT is there and WHERE it is