
---

# Vision Transformer (ViT) for Image Classification

## Introduction to Vision Transformer

The **Vision Transformer (ViT)** was introduced by researchers at Google Research in their paper titled **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**, published in October 2020. ViT is a novel architecture that leverages the power of the Transformer model, traditionally used for Natural Language Processing (NLP), to process and classify images, which are typically handled by convolutional neural networks (CNNs). The model achieved competitive performance on image classification tasks with significantly less computational cost when pre-trained on large datasets.



---

## Background and Motivation

### 1. **Dominance of CNNs in Vision Tasks**
   Convolutional Neural Networks (CNNs) have been the go-to architecture for image classification and computer vision tasks due to their ability to capture local spatial features effectively using convolutional layers. However, CNNs have limitations in modeling global context due to their inherent design of local receptive fields.

### 2. **Introduction of Transformers in NLP**
   Transformers revolutionized the field of NLP by replacing recurrent architectures (RNNs, LSTMs) with self-attention mechanisms that capture long-range dependencies more efficiently. This prompted researchers to investigate whether similar benefits could be achieved by applying transformers to vision tasks.

### 3. **Motivation for Vision Transformers**
   The primary motivation behind ViT was to explore whether the highly parallelized self-attention mechanism used in NLP could outperform traditional CNNs by modeling global context more efficiently for image classification tasks. Additionally, ViT eliminates the need for convolutions by dividing an image into patches, treating them as tokens, and applying the transformer mechanism directly to these patches.

---

## Key Innovation

The **key innovation** of ViT lies in **adapting the Transformer architecture to image data**. Instead of using convolutions to process images, ViT splits an image into fixed-sized patches, linearly embeds each patch, and treats them as a sequence of tokens, similar to the way words are treated in NLP tasks. By doing so, it utilizes the self-attention mechanism of transformers to capture both local and global dependencies in the image more effectively.

Key breakthroughs include:
1. **Patch-based Image Embedding**: Images are split into patches, reducing the complexity compared to processing individual pixels directly.
2. **Global Self-Attention**: ViT uses self-attention to capture relationships between patches, making it capable of handling long-range dependencies between image regions.
3. **Scalability**: ViT scales well when pre-trained on large datasets like JFT-300M and fine-tuned on smaller datasets, outperforming CNNs like ResNet.

---

## Architecture of Vision Transformer (ViT)

![alt text](<ViT Architecture.png>)

The ViT architecture closely mirrors the original transformer design used in NLP tasks, with a few modifications for handling image data. Here’s a breakdown of the key components:

### 1. **Patch Embedding Layer**  
   - The image is divided into non-overlapping patches of a fixed size, such as 16x16 pixels.  
   - Each patch is flattened into a vector, and a **linear projection** is applied to map it to a latent space.  
   - These patch embeddings, along with a **class token** (used for classification), form the input to the transformer.

### 2. **Positional Embedding**  
   - Since transformers do not inherently capture the order of the input sequence, **positional embeddings** are added to the patch embeddings.  
   - These positional embeddings provide information about the position of each patch within the image, ensuring that spatial relationships are preserved.

### 3. **Transformer Encoder**  
   - The core of the Vision Transformer is the **transformer encoder**, which consists of multiple **Transformer Encoder Layers** stacked on top of each other.
   - Each encoder layer has two primary components:
     - **Multi-Head Self-Attention (MHSA)**: This captures global dependencies by computing attention scores between all patches.
     - **Feed-Forward Neural Network (FFN)**: A fully connected network applied to each token after self-attention.
   - **Layer normalization** and **residual connections** are used for stability and faster convergence.

### 4. **Classification Token (CLS Token)**  
   - The class token is prepended to the sequence of patch embeddings and interacts with the rest of the tokens during the transformer layers.
   - After passing through the transformer, the class token contains information that is used for classification.

### 5. **MLP Head for Classification**  
   - After processing through the transformer, the output corresponding to the class token is passed through a **multi-layer perceptron (MLP)** head, which generates the final class probabilities.

---

## Flow of Vision Transformer Model (ViT)

The flow of the ViT model can be broken down as follows:

1. **Input Image Processing**:
   - The input image is split into fixed-sized patches (e.g., 16x16). Each patch is flattened and projected into a higher-dimensional space using a linear layer.

2. **Add Class Token & Positional Embeddings**:
   - A special class token (CLS) is added to the beginning of the sequence. Positional embeddings are added to the patch embeddings to retain the spatial structure of the image.

3. **Transformer Encoder**:
   - The sequence of embeddings is passed through multiple transformer encoder layers, which consist of:
     - **Multi-Head Self-Attention**: Computes attention scores across patches to capture long-range dependencies.
     - **Feed-Forward Neural Network**: Applied to each token, helping model complex relationships.
     - **Residual Connections**: Skip connections are added to stabilize the network.

4. **Classification Head**:
   - After the final transformer encoder layer, the class token contains information about the entire image. This token is passed through a fully connected MLP head, which outputs class probabilities for image classification.

5. **Prediction**:
   - The output from the classification head is used to predict the class of the input image.

---

## Conclusion

The **Vision Transformer (ViT)** represents a shift in how we approach image classification tasks. By applying transformers to images in a patch-wise manner, ViT removes the need for convolutions, allowing for a model that scales well with large datasets and achieves competitive performance compared to CNNs. Its ability to model global context efficiently through self-attention is a major step forward in the field of computer vision.
