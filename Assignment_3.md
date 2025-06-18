# Assignment 3

## Question 1

### Why do neural networks need activation functions?

Neural networks need activation functions to introduce non-linearity into the model. Without activation functions, the network would simply be a linear transformation (a series of matrix multiplications), no matter how many layers it has, and thus could only learn linear relationships. Activation functions like ReLU, sigmoid, or tanh enable neural networks to approximate complex, non-linear functions, making them capable of learning from highly intricate data patterns (e.g., images, speech, etc.).

### Briefly describe the influence of the value of the learning rate on the training of the neural network.

The learning rate controls the step size taken during gradient descent optimization:

* **Too high**: The model may overshoot optimal solutions, causing unstable training (divergence or oscillations).
* **Too low**: Training becomes very slow, and the model might get stuck in poor local minima or plateaus.
An optimal learning rate balances speed and stability, ensuring efficient convergence. Techniques like learning rate schedules or adaptive optimizers (e.g., Adam) help manage this.

### What advantages does CNN have over fully connected DNN in image classification?

CNNs outperform fully connected DNNs in image classification due to:

* **Local connectivity**: CNNs use kernels/filters that focus on local regions (e.g., edges, textures), capturing spatial hierarchies.
* **Parameter efficiency**: Weight sharing in convolutions drastically reduces parameters compared to dense layers, lowering overfitting risk.
* **Translation invariance**: Pooling and convolutions make CNNs robust to shifts/rotations in input images.
* **Hierarchical feature learning**: Early layers detect low-level features (edges), while deeper layers combine them into high-level concepts (e.g., objects).

These properties make CNNs faster, more accurate, and scalable for image data.

## Question 2

**Given:**

* Input size: $227 \times 227 \times 3$ (width $\times$ height $\times$ channels)
* CONV1 layer:
  * Number of filters ($K$): $96$
  * Filter size ($F \times F$): $11 \times 11$
  * Stride ($S$): $4$
  * Padding ($P$): $0$

**Output width/height ($W_{out}$) formula:**

$$
W_{out} = \left\lfloor \frac{W_{in} - F + 2P}{S} \right\rfloor + 1
$$

**Calculation:**

1. **Width/Height dimension:**

   $$
   W_{out} = \left\lfloor \frac{227 - 11 + 2 \times 0}{4} \right\rfloor + 1 = \left\lfloor \frac{216}{4} \right\rfloor + 1 = 54 + 1 = 55
   $$

2. **Depth dimension:** Equal to the number of filters ($K = 96$).

**Final output size:**

$55 \times 55 \times 96$

## Question 3

### Convolution Operations: 4×4 Feature Map with 3×3 Kernel (Stride=1)

**Given:**

* **Feature Map ($I$):**

  $$
  \begin{bmatrix}
  1 & 2 & 3 & 0 \\
  0 & 1 & 2 & 3 \\
  3 & 0 & 1 & 2 \\
  2 & 3 & 0 & 1 \\
  \end{bmatrix}
  $$

* **Kernel ($K$):**

  $$
  \begin{bmatrix}
  2 & 0 & 1 \\
  0 & 1 & 2 \\
  1 & 0 & 2 \\
  \end{bmatrix}
  $$

* **Stride ($S$):** $1$

#### (1) No Padding ($P=0$)

**Output size formula:**

$$
W_{out} = \left\lfloor \frac{W_{in} - F + 2P}{S} \right\rfloor + 1 = \left\lfloor \frac{4 - 3 + 0}{1} \right\rfloor + 1 = 2
$$

**Output feature map size:** $2 \times 2$

**Calculation (Element-wise multiplication + Sum):**

1. **Top-left corner ($O_{1,1}$):**

   $$
   \begin{bmatrix}
   1 & 2 & 3 \\
   0 & 1 & 2 \\
   3 & 0 & 1 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 2+0+3+0+1+4+3+0+2 = 15
   $$

2. **Top-right corner ($O_{1,2}$):**

   $$
   \begin{bmatrix}
   2 & 3 & 0 \\
   1 & 2 & 3 \\
   0 & 1 & 2 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 4+0+0+0+2+6+0+0+4 = 16
   $$

3. **Bottom-left corner ($O_{2,1}$):**

   $$
   \begin{bmatrix}
   0 & 1 & 2 \\
   3 & 0 & 1 \\
   2 & 3 & 0 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+2+0+0+2+2+0+0 = 6
   $$

4. **Bottom-right corner ($O_{2,2}$):**

   $$
   \begin{bmatrix}
   1 & 2 & 3 \\
   0 & 1 & 2 \\
   3 & 0 & 1 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 2+0+3+0+1+4+3+0+2 = 15
   $$

**Output without padding:**

$$
\begin{bmatrix}
15 & 16 \\
6 & 15 \\
\end{bmatrix}
$$

#### (2) With Padding ($P=1$ to maintain input size)

**Padded Feature Map ($I_{\text{padded}}$):**

$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 2 & 3 & 0 & 0 \\
0 & 0 & 1 & 2 & 3 & 0 \\
0 & 3 & 0 & 1 & 2 & 0 \\
0 & 2 & 3 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

**Output size:** $4 \times 4$ (same as input).

**Calculation (Key positions):**

1. **$O_{1,1}$ (Top-left):**

   $$
   \begin{bmatrix}
   0 & 0 & 0 \\
   0 & 1 & 2 \\
   0 & 0 & 1 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+0+0+1+4+0+0+2 = 7
   $$

2. **$O_{1,2}$:**

   $$
   \begin{bmatrix}
   0 & 0 & 0 \\
   1 & 2 & 3 \\
   0 & 1 & 2 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+0+0+2+6+0+0+4 = 12
   $$

3. **$O_{1,3}$:**

   $$
   \begin{bmatrix}
   0 & 0 & 0 \\
   2 & 3 & 0 \\
   1 & 2 & 3 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+0+0+3+0+1+0+6 = 10
   $$

4. **$O_{1,4}$ (Top-right):**

   $$
   \begin{bmatrix}
   0 & 0 & 0 \\
   3 & 0 & 0 \\
   2 & 3 & 0 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+0+0+0+0+2+0+0 = 2
   $$

5. **$O_{2,1}$:**

   $$
   \begin{bmatrix}
   0 & 1 & 2 \\
   0 & 0 & 1 \\
   0 & 3 & 0 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+2+0+0+2+0+0+0 = 4
   $$

6. **$O_{2,2}$:**

   $$
   \begin{bmatrix}
   1 & 2 & 3 \\
   0 & 1 & 2 \\
   3 & 0 & 1 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 2+0+3+0+1+4+3+0+2 = 15
   $$

7. **$O_{2,3}$:**

   $$
   \begin{bmatrix}
   2 & 3 & 0 \\
   1 & 2 & 3 \\
   0 & 1 & 2 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 4+0+0+0+2+6+0+0+4 = 16
   $$

8. **$O_{2,4}$:**

   $$
   \begin{bmatrix}
   3 & 0 & 0 \\
   2 & 3 & 0 \\
   1 & 2 & 0 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 6+0+0+0+3+0+1+0+0 = 10
   $$

9. **$O_{3,1}$:**

   $$
   \begin{bmatrix}
   0 & 0 & 1 \\
   0 & 3 & 0 \\
   0 & 2 & 3 \\
   \end{bmatrix}
   \odot
   \begin{bmatrix}
   2 & 0 & 1 \\
   0 & 1 & 2 \\
   1 & 0 & 2 \\
   \end{bmatrix}
   = 0+0+1+0+3+0+0+0+6 = 10
   $$

10. **$O_{3,2}$:**

    $$
    \begin{bmatrix}
    0 & 1 & 2 \\
    3 & 0 & 1 \\
    2 & 3 & 0 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 0+0+2+0+0+2+2+0+0 = 6
    $$

11. **$O_{3,3}$:**

    $$
    \begin{bmatrix}
    1 & 2 & 3 \\
    0 & 1 & 2 \\
    3 & 0 & 1 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 2+0+3+0+1+4+3+0+2 = 15
    $$

12. **$O_{3,4}$:**

    $$
    \begin{bmatrix}
    2 & 3 & 0 \\
    1 & 2 & 0 \\
    0 & 1 & 0 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 4+0+0+0+2+0+0+0+0 = 6
    $$

13. **$O_{4,1}$ (Bottom-left):**

    $$
    \begin{bmatrix}
    0 & 3 & 0 \\
    0 & 2 & 3 \\
    0 & 0 & 0 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 0+0+0+0+2+6+0+0+0 = 8
    $$

14. **$O_{4,2}$:**

    $$
    \begin{bmatrix}
    3 & 0 & 1 \\
    2 & 3 & 0 \\
    0 & 0 & 0 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 6+0+1+0+3+0+0+0+0 = 10
    $$

15. **$O_{4,3}$:**

    $$
    \begin{bmatrix}
    0 & 1 & 2 \\
    3 & 0 & 1 \\
    0 & 0 & 0 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 0+0+2+0+0+2+0+0+0 = 4
    $$

16. **$O_{4,4}$ (Bottom-right):**

    $$
    \begin{bmatrix}
    1 & 2 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 0 \\
    \end{bmatrix}
    \odot
    \begin{bmatrix}
    2 & 0 & 1 \\
    0 & 1 & 2 \\
    1 & 0 & 2 \\
    \end{bmatrix}
    = 2+0+0+0+1+0+0+0+0 = 3
    $$

**Output with padding:**

$$
\begin{bmatrix}
7 & 12 & 10 & 2 \\
4 & 15 & 16 & 10 \\
10 & 6 & 15 & 6 \\
8 & 10 & 4 & 3 \\
\end{bmatrix}
$$

### Pooling Operations: 4×4 Feature Map with 2×2 Pooling (Stride=2)

**Given Feature Map ($I$):**

$$
\begin{bmatrix}
1 & 4 & 2 & 1 \\
5 & 8 & 3 & 4 \\
7 & 6 & 4 & 5 \\
1 & 3 & 1 & 2 \\
\end{bmatrix}
$$

#### (1) Max-Pooling

**Output size:** $\left\lfloor \frac{4 - 2}{2} \right\rfloor + 1 = 2 \times 2$

**Calculations:**

1. **Top-left window:**

   $$
   \begin{bmatrix}
   1 & 4 \\
   5 & 8 \\
   \end{bmatrix}
   \implies \max(1, 4, 5, 8) = 8
   $$

2. **Top-right window:**

   $$
   \begin{bmatrix}
   2 & 1 \\
   3 & 4 \\
   \end{bmatrix}
   \implies \max(2, 1, 3, 4) = 4
   $$

3. **Bottom-left window:**

   $$
   \begin{bmatrix}
   7 & 6 \\
   1 & 3 \\
   \end{bmatrix}
   \implies \max(7, 6, 1, 3) = 7
   $$

4. **Bottom-right window:**

   $$
   \begin{bmatrix}
   4 & 5 \\
   1 & 2 \\
   \end{bmatrix}
   \implies \max(4, 5, 1, 2) = 5
   $$

**Max-Pooling Output:**

$$
\begin{bmatrix}
8 & 4 \\
7 & 5 \\
\end{bmatrix}
$$

#### (2) Average-Pooling

**Output size:** $\left\lfloor \frac{4 - 2}{2} \right\rfloor + 1 = 2 \times 2$

**Calculations:**

1. **Top-left window:**

   $$
   \frac{1 + 4 + 5 + 8}{4} = \frac{18}{4} = 4.5
   $$

2. **Top-right window:**

   $$
   \frac{2 + 1 + 3 + 4}{4} = \frac{10}{4} = 2.5
   $$

3. **Bottom-left window:**

   $$
   \frac{7 + 6 + 1 + 3}{4} = \frac{17}{4} = 4.25
   $$

4. **Bottom-right window:**

   $$
   \frac{4 + 5 + 1 + 2}{4} = \frac{12}{4} = 3
   $$

**Average-Pooling Output:**

$$
\begin{bmatrix}
4.5 & 2.5 \\
4.25 & 3 \\
\end{bmatrix}
$$
