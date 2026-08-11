# Backpropagation and Gradient Descent — A Simple Example

This document explains **backpropagation** and **gradient descent** using the
smallest possible neural network and a single training example.

---

## 1. Example #1

x ──(w)──> z ──σ──> ŷ


- `x` : input  
- `w` : weight  
- `σ` : sigmoid activation function  
- `ŷ` : predicted output  

---

## 2. Given Values

### Input
x = 1.0

### True Label
y = 0


### Initial Weight
w = 0.5

### Learning Rate
η = 0.1

---

## 3. Activation and Loss Functions

### Sigmoid Activation Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Loss Function (Mean Squared Error)

$$
L = (\hat{y} - y)^2
$$

---

## 4. Forward Pass

### Step 1: Linear Transformation

$$
z = w \cdot x = 0.5 \times 1.0 = 0.5
$$

### Step 2: Apply Activation Function

$$
\hat{y} = \sigma(0.5) \approx 0.622
$$

---

## 5. Compute Loss

$$
L = (0.622 - 0)^2 \approx 0.387
$$

The prediction is higher than the true label, resulting in a positive loss.

---

## 6. Backpropagation (Gradient Computation)

The goal of backpropagation is to compute the gradient of the loss with respect
to the weight:

$$
\frac{\partial L}{\partial w}
$$

Using the **chain rule**:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial z}
\cdot
\frac{\partial z}{\partial w}
$$

---

### Step 6.1: Gradient of Loss w.r.t. Prediction

$$
\frac{\partial L}{\partial \hat{y}}
= 2(\hat{y} - y)
= 2(0.622 - 0)
= 1.244
$$

---

### Step 6.2: Gradient of Sigmoid

$$
\frac{\partial \hat{y}}{\partial z}
= \sigma(z)\bigl(1 - \sigma(z)\bigr)
$$

$$
= 0.622 \times (1 - 0.622)
\approx 0.235
$$

---

### Step 6.3: Gradient of Linear Function

$$
\frac{\partial z}{\partial w} = x = 1.0
$$

---

### Step 6.4: Final Gradient

$$
\frac{\partial L}{\partial w}
= 1.244 \times 0.235 \times 1.0
\approx 0.292
$$

This value represents how much the loss changes when the weight changes.

---

## 7. Gradient Descent (Weight Update)

Gradient descent updates the weight using:

$$
w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w}
$$

Substituting values:

$$
w_{\text{new}} = 0.5 - 0.1 \times 0.292
= 0.4708
$$

---

## 8. Verify Improvement

### New Prediction

$$
\hat{y} = \sigma(0.4708) \approx 0.615
$$

### New Loss

$$
L = (0.615 - 0)^2 \approx 0.378
$$

The loss has decreased, confirming that the update improved the model.

---

## Example 2

A single neuron with two inputs and a bias:

- Network: x1, x2 ──(w1,w2,b)──> z ──σ──> ŷ
- Given: x1 = 1.0, x2 = 2.0, y = 1  
  w1 = 0.5, w2 = -0.5, b = 0.0, η = 0.1

Forward pass:

$$
z = w_1 x_1 + w_2 x_2 + b = 0.5\cdot1 + (-0.5)\cdot2 + 0 = -0.5
$$

$$
\hat{y} = \sigma(-0.5) \approx 0.3775
$$

$$
L = (\hat{y} - y)^2 \approx 0.387
$$

Gradients (chain rule):

$$
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y}-y) \approx -1.2449
$$

$$
\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y}) \approx 0.2350
$$

$$
\frac{\partial L}{\partial z} \approx -0.292
$$

Parameter gradients:

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z}\,x_1 \approx -0.292
$$

$$
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z}\,x_2 \approx -0.584
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \approx -0.292
$$

Gradient descent update:

$$
w_1' = 0.5 - 0.1(-0.292) = 0.5292
$$

$$
w_2' = -0.5 - 0.1(-0.584) = -0.4416
$$

$$
b' = 0 - 0.1(-0.292) = 0.0292
$$

Verify improvement:

$$
z' \approx 0.5292\cdot1 + (-0.4416)\cdot2 + 0.0292 \approx -0.3248
$$

$$
\hat{y}' = \sigma(-0.3248) \approx 0.4196
$$

$$
L' = (0.4196 - 1)^2 \approx 0.338
$$

Loss decreased after one update.

---

## Example 3: Batch vs Stochastic Gradient Descent (3 data points)

Use the simple single-weight model from Example #1: $z = w x, ŷ = σ(z)$. Start with $w = 0.5$ and $η = 0.1$.

Dataset (3 points):
- (x1,y1) = (1, 0)
- (x2,y2) = (2, 1)
- (x3,y3) = (3, 1)

Compute predictions at w = 0.5:

$$
\hat{y}_1 \approx 0.622,\quad \hat{y}_2 \approx 0.731,\quad \hat{y}_3 \approx 0.818
$$

Per-sample gradients (at w = 0.5):

$$
g_1 \approx 0.292,\quad g_2 \approx -0.211,\quad g_3 \approx -0.163
$$

Batch (full-dataset) gradient = average:

$$
g_{\text{batch}} = \frac{g_1+g_2+g_3}{3} \approx -0.0275
$$

Batch update:

$$
w_{\text{batch}} = 0.5 - 0.1\cdot(-0.0275) \approx 0.5028
$$

Stochastic (online) — apply one-sample update sequentially, recomputing gradients after each update:

- After sample 1: w ← 0.5 - 0.1·0.292 = 0.4708
- Recompute gradient for sample 2 at w=0.4708 → update → w ≈ 0.4933
- Recompute gradient for sample 3 at w≈0.4933 → update → w ≈ 0.5102

Final weights after one pass:

$$
w_{\text{batch}}\approx 0.5028,\quad w_{\text{stochastic}}\approx 0.5102
$$

Takeaway: with the same initial w and learning rate, batch (average) update and one-pass stochastic updates can move w differently; stochastic updates follow each sample's gradient immediately and can produce larger cumulative changes in one pass.

---

## 9. Key Takeaways

- **Backpropagation**
  - Computes gradients using the chain rule
  - Determines how each weight affects the loss
  - Works on a single data point or a batch

- **Gradient Descent**
  - Uses gradients to update weights
  - Moves weights in the direction that reduces loss

---

## Summary

**Backpropagation computes gradients.  
Gradient descent uses those gradients to update weights and learn.**
