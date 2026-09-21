# directional Top K SAE  

we add a direction alignment score to the total criteria for picking the top K  

the problem with TopK SAE is very big encoders always get picked over smaller encoders that are directionally closer to the activation. causing many dead features and defeating the purpose of the SAE which is to be a activation -> concept solver.  

we will use the cosine similarity either as the pure criteria or part of it as the picker for TopK instead.  

we calculate a topK_score to decide topK picks by. in traditional this is just the activation score.  

# pure directional top k SAE  

topK_score = encoder_direction @ activation / encoder_direction.norm() / activation.norm()  

# directional and value top K SAE  

topK_score = directional_score * directional_coefficient + activation_value  

we hypersearch directional_coefficient.   
