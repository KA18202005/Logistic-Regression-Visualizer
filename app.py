import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        return X, y

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.05)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.05)

    XX, YY = np.meshgrid(a, b)
    input_array = np.c_[XX.ravel(), YY.ravel()]
    return XX, YY, input_array

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox('Select Dataset', ('Binary', 'Multiclass'))

penalty = st.sidebar.selectbox(
    'Regularization',
    ('l2', 'l1', 'elasticnet', 'none')
)

solver = st.sidebar.selectbox(
    'Solver',
    ('lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag')
)

c_input = st.sidebar.number_input('C', value=1.0)
max_iter = st.sidebar.number_input('Max Iterations', value=200)

multi_class = st.sidebar.selectbox(
    'Multi Class',
    ('auto', 'ovr', 'multinomial')
)

l1_ratio = st.sidebar.slider('l1 Ratio (only for elasticnet)', 0.0, 1.0, 0.5)

# Plot initial data
fig, ax = plt.subplots()
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    # ----- VALIDATION LOGIC -----
    if penalty == 'elasticnet' and solver != 'saga':
        st.error("ElasticNet penalty only works with saga solver")
        st.stop()

    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
        st.error("L1 penalty only works with liblinear or saga solver")
        st.stop()

    if solver == 'liblinear' and multi_class == 'multinomial':
        st.error("liblinear does not support multinomial")
        st.stop()

    if penalty != 'elasticnet':
        l1_ratio = None

    clf = LogisticRegression(
        penalty=penalty,
        C=c_input,
        solver=solver,
        max_iter=max_iter,
        multi_class=multi_class,
        l1_ratio=l1_ratio
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)

    fig, ax = plt.subplots()
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolor='black')
    ax.set_xlabel("Col1")
    ax.set_ylabel("Col2")

    st.pyplot(fig)
    st.subheader("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 2)))
