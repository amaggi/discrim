import os
from syn_catalog import *

# Trimouns
filename='trimouns.hdf5'
#eq_cat=generate_eq_catalog(1000,-30,30,-30,30)
#blast_cat=generate_blast_catalog(1000,0,0,3,3,30,17,0.1)
#write_catalogs_hdf5(eq_cat, blast_cat, filename)
#eq, blast = read_catalogs_hdf5(filename)
#plot_catalogs(eq,blast)

# Mollard
filename='mollard.hdf5'
eq_cat=generate_eq_catalog(1000,-40,40,-40,40)
blast_cat=generate_blast_catalog(1000,0,0,10,10,30,12,2)
write_catalogs_hdf5(eq_cat, blast_cat, filename)
eq, blast = read_catalogs_hdf5(filename)
plot_catalogs(eq,blast)
