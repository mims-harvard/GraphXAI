import numpy as np
import torch
from sklearn.utils.random import sample_without_replacement

from graphxai.utils import check_random_state


def _generate_hypercube(samples, dimensions, rng):
    """
    Returns distinct binary samples of length dimensions.
    """
    if dimensions > 30:
        return np.hstack([rng.randint(2, size=(samples, dimensions - 30)),
                          _generate_hypercube(samples, 30, rng)])
    out = sample_without_replacement(2 ** dimensions, samples,
                                     random_state=rng).astype(dtype='>u4', copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out


def make_structured_feature(y: torch.Tensor, n_features=5, n_informative=2,
                            n_redundant=0, n_repeated=0, n_clusters_per_class=2,
                            unique_explanation=True, flip_y=0.01,
                            class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                            shuffle=True, seed=None):
    """This function is based on sklearn.datasets.make_classification.

    Generate structured features for the given labels.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.

    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, default=2
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, default=2
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, default=0
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_clusters_per_class : int, default=2
        The number of clusters per class.

    flip_y : float, default=0.01
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than ``n_classes`` in y in some cases.

    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    seed : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See Also
    --------
    make_blobs : Simplified variant.
    make_multilabel_classification : Unrelated rng for multilabel tasks.
    """

    # print('class sep', class_sep)
    # print('n_informative', n_informative)

    Yorg = y.clone().numpy()

    if isinstance(y, torch.Tensor):
        y = y.clone().numpy()

    n_samples = y.shape[0]
    labels, n_samples_per_class = np.unique(y, return_counts=True)
    n_classes = len(labels)

    rng = check_random_state(seed)

    # Set n_redundant and n_repeated to 0 if unique explanation
    if unique_explanation:
        n_redundant = n_repeated = 0

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(msg.format(n_classes, n_clusters_per_class,
                                    n_informative, 2**n_informative))

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters
    n_samples_per_cluster = [
        int(n_samples_per_class[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X
    X = np.zeros((n_samples, n_features))

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative,
                                    rng).astype(float, copy=False)

    # print('centroids', centroids)
    # print('n_clusters', n_clusters)
    # print('n_classes', n_classes)

    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= rng.rand(n_clusters, 1)
        centroids *= rng.rand(1, n_informative)

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = rng.randn(n_samples, n_informative)

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster

        A = 2 * rng.rand(n_informative, n_informative) - 1
        X_k[...] = np.dot(X_k, A)  # introduce random covariance

        X_k += centroid  # shift the cluster to a vertex
        #print('k', k)

    # Create redundant features
    if n_redundant > 0:
        B = 2 * rng.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * rng.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = rng.randn(n_samples, n_useless)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = rng.rand(n_samples) < flip_y
        y[flip_mask] = rng.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * rng.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * rng.rand(n_features)
    X *= scale

    # The binary feature mask (1 for informative features) if unique explanation
    if unique_explanation:
        feature_mask = np.zeros(n_features, dtype=bool)
        feature_mask[:n_informative] = True

    #print('y before shuffle', y)

    if shuffle:
        # Randomly permute features
        indices = np.arange(n_features)
        rng.shuffle(indices)
        X[:, :] = X[:, indices]
        #y = np.array([y[i] for i in indices])
        if unique_explanation:
            feature_mask[:] = feature_mask[indices]

    #print('y', list(y))
    # Sort y and then assign to each spot in the tensor y
    #unique_y = torch.sort(torch.unique(Yorg))

    unique_y = np.sort(np.unique(Yorg))

    #print('y', y)

    #print(unique_y)
    #ysort = np.sort(y)

    Xnew = np.zeros_like(X)
    
    for yval in unique_y:
        ingenerated = np.argwhere(y == yval).flatten()
        #print('ingenerated', ingenerated)
        inorg = np.argwhere(Yorg == yval).flatten()
        #print('inorg', inorg)

        for gen, org in zip(ingenerated, inorg): # Move 
            Xnew[org, :] = X[gen, :]

    #print(Xnew[:10, :])

    # Convert to tensor
    Xnew = torch.from_numpy(Xnew).float()
    feature_mask = torch.from_numpy(feature_mask)

    if unique_explanation:
        return Xnew, feature_mask
    else:
        return Xnew
