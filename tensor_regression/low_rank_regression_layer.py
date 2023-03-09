import matplotlib.pyplot as plt
from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.datasets.synthetic import gen_image
from tensorly.regression import CPRegressor
import tensorly.backend as T
import tensorly as tl

# Parameter of experiment
image_width = 25
image_height = 25

# fix the random seed for reproducibility
rng = tl.check_random_state(1)

# Generate a random tensor
X = T.tensor(rng.normal(size=(1000, image_height, image_width), loc=0, scale=1))

# Generate the original image
weight_img = gen_image(region='swiss', image_height=image_height, image_width=image_width)
weight_img = T.tensor(weight_img)

# The true labels is obtained by taking the product between the true regression weights and the input tensors
y = T.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(weight_img))

# Let's view the true regression weight
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.imshow(T.to_numpy(weight_img), cmap=plt.cm.OrRd, interpolation='nearest')
axis.set_axis_off()
axis.set_title('True regresison weights')

# Create a tensor regressor estimator
estimator = CPRegressor(weight_rank=2, tol=10e-7, n_iter_max=100, reg_W=1, verbose=0)

# Fit the estimator to the data
estimator.fit(X, y)

# Predict the labels given input tensors
print(estimator.predict(X))

# Visualize the learned weights
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.imshow(T.to_numpy(estimator.weight_tensor_), cmap=plt.cm.OrRd, interpolation='nearest')
axis.set_axis_off()
axis.set_title('Learned regresison weights')

# Access the decomposed form below
weights, factors = estimator.cp_weight_