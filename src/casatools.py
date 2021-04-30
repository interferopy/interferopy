

# Write a CASA mask region file using a list of coordinates (degrees) and circle sizes (radius in arcses)
def casareg(outfile, ra, dec, rad):
    f = open(outfile, "w")
    f.write("#CRTFv0 CASA Region Text Format version 0 \n")
    for i in range(len(ra)):
        f.write("circle[[" + str(ra[i]) + "deg," + str(dec[i]) + "deg], " + str(rad[i]) + "arcsec]" + '\n')
    f.close()
