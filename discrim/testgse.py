from catalog_io_plot import read_gse2_cat, read_gse2_cat_from_directory, write_catalog_hdf5

# test gse 
filename = "/home/nabil/discrim-master/CATALOGS/ldg_all/P2012003.cl1.txt"
X, y = read_gse2_cat(filename)

X, y = read_gse2_cat_from_directory("/home/nabil/discrim-master/CATALOGS/ldg_all")
print X.shape, y.shape

write_catalog_hdf5(X, y, 'ldg.hdf5')
