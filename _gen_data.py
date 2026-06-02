"""Quick data generation script for local run."""
import sys
sys.path.insert(0, '.')
from src.data.pipeline import (
    SupplyChainNetworkGenerator, TradefinanceDataGenerator,
    TimeSeriesGenerator, SyntheticDataGenerator
)
from pathlib import Path

raw = Path('data/raw')
raw.mkdir(parents=True, exist_ok=True)

print('  Generating 500 suppliers...')
gen = SupplyChainNetworkGenerator(seed=42)
sup = gen.generate_suppliers(n=500)
sup.to_csv(raw / 'suppliers_500.csv', index=False)
print(f'    OK  suppliers_500.csv  {sup.shape}')

print('  Generating 2000 supply chain edges...')
edges = gen.generate_edges(sup, n_edges=2000)
edges.to_csv(raw / 'supply_chain_edges_2000.csv', index=False)
print(f'    OK  supply_chain_edges_2000.csv  {edges.shape}')

print('  Generating 5000 LC transactions...')
tf = TradefinanceDataGenerator(seed=42)
lc = tf.generate_lc_transactions(n=5000)
lc.to_csv(raw / 'lc_transactions_5000.csv', index=False)
dr = lc['default_flag'].mean()
print(f'    OK  lc_transactions_5000.csv  {lc.shape}  default_rate={dr:.2%}')

print('  Generating 500 SCF invoices...')
scf = tf.generate_scf_invoices(n=500)
scf.to_csv(raw / 'scf_invoices_500.csv', index=False)
print(f'    OK  scf_invoices_500.csv  {scf.shape}')

print('  Generating carriers + shipments + financial data...')
sg = SyntheticDataGenerator(seed=42)
data = sg.generate_all(save_path='data/raw')
for k, v in data.items():
    print(f'    OK  {k}.csv  {v.shape}')

print()
print('All synthetic datasets generated!')
print(f'Files saved to: {raw.resolve()}')
