"""Reproduce the audited 2024 Wayanad imagery stack."""
from trustsr.fetch import fetch


if __name__ == '__main__':
    output = fetch((76.233, 11.782, 10000), '2024-01-01T00:00:00Z', '2024-12-31T23:59:59Z',
                   audit=('risk/results/r2_catalog.json', 'risk/results/r2_acquisitions.json'))
    print(output)
