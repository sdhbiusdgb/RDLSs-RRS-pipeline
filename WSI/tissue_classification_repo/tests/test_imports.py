def test_imports():
    import tissue_classifier  # noqa
    from tissue_classifier.data.dataset import TIFDataset  # noqa
    from tissue_classifier.models.resnet50_ccl import ResNet50CCLFeatureExtractor  # noqa
    from tissue_classifier.models.factory import build_tissue_classifier  # noqa
