try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='federated_gbdt',
    version='1.0.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=[
        'federated_gbdt',
        'federated_gbdt.models',
        'federated_gbdt.core',
        'federated_gbdt.models.base',
        'federated_gbdt.models.gbdt',
        'federated_gbdt.models.gbdt.components',
        'federated_gbdt.core.binning',
        'federated_gbdt.core.dp_multiq',
        'federated_gbdt.core.moments_accountant',
        'federated_gbdt.core.pure_ldp',
        'federated_gbdt.core.pure_ldp.core',
        'federated_gbdt.core.pure_ldp.frequency_oracles',
        'federated_gbdt.core.pure_ldp.frequency_oracles.square_wave',
        'federated_gbdt.core.pure_ldp.frequency_oracles.local_hashing',
        'federated_gbdt.core.pure_ldp.frequency_oracles.hybrid_mechanism',
    ],
)
