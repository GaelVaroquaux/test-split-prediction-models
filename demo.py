
import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator, for reproducibility
np.random.seed(0)


# The (unknown) prediction mechanism
def f(x, noise=.5):
    return x - .1*x**2 - x**3 + .1*x**5 + noise*np.random.normal(size=x.shape)


# A polynomial regression model
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

model = make_pipeline(PolynomialFeatures(5), LinearRegression())

# Some data
x = np.random.uniform(low=-1, high=1, size=20)
y = f(x)


plt.figure(figsize=(3, 2.5))

# Plot the data points
plt.plot(x, y, 'kx', markersize=10, label='Data used for model fit',
         zorder=100)

# Plot the model predictions
x_test = np.linspace(-1, 1, 100)
model.fit(x.reshape((-1, 1)), y)
plt.plot(x_test, model.predict(x_test.reshape((-1, 1))), color='C0',
         label='Model predictions')

plt.plot(x_test, f(x_test, noise=0), color='.9', zorder=0)

# New data (generalization error)
x_new = np.random.uniform(low=-1, high=1, size=200)
y_new = f(x_new)

plt.plot(x_new, y_new, '.', color='k', alpha=.2, markersize=5,
         label='New data (generalization)', mec='none',
         zorder=0)

# Style the figure
plt.xticks((-1, 1), size=6)
plt.yticks(())
plt.axis('tight')
plt.legend(loc='best', frameon=False)
plt.xlabel('Measurement', labelpad=-8)
plt.ylabel('Predicted outcome')

ax = plt.gca()
for spine in ['right', 'top']:
    spine = ax.spines[spine]
    spine.set_visible(False)

plt.tight_layout(pad=.1)
plt.savefig('generalization_error.pdf')
plt.savefig('generalization_error.png')
