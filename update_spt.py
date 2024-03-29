import pyfits, commands, os, time, numpy, math
from numpy import asarray as ar
from scipy.interpolate import interp1d

def get_ephem( ):

    FILE = open('/net/frakir/home/aparker/hubble_orbit_ephemeris_full_jd.asc', 'r')
    dat  = FILE.readlines()
    FILE.close()
    
    JD, X,Y,Z = [],[],[],[]
    for k in dat:
        if k.strip() != '':
            l = numpy.asarray(k.strip().split(), dtype='float')
            #print l[0]
            JD.append( l[0] )
            X.append(  l[1] )
            Y.append(  l[2] )
            Z.append(  l[3] )


    JD, X, Y, Z = ar(JD), ar(X), ar(Y), ar(Z)
    
    Xi = interp1d( JD, X, kind='linear' )
    Yi = interp1d( JD, Y, kind='linear' )        
    Zi = interp1d( JD, Z, kind='linear'  )

    print min(JD), max(JD)
    
    return Xi, Yi, Zi, min(JD), max(JD)

def new_spt():
    fn = commands.getoutput('ls *_flt.fits').split('\n')
    dx = []
    km2AU = 1.0 / 1.49598e8

    Xi, Yi, Zi, minJD, maxJD =  get_ephem( )

    for k in fn:
        #keys = commands.getoutput('gethead %s POSTNSTX POSTNSTY POSTNSTZ PSTRTIME'%(k)).split()
        #keys2 = commands.getoutput('gethead %s_flt.fits EXPSTART'%(k.split('_')[0])).split()
        #keys2 = commands.getoutput('gethead %s_flt.fits EXPSTART EXPEND'%(k.split('_')[0])).split()

        h0 = pyfits.open('%s_flt.fits'%(k.split('_')[0]))
        header0 = h0[0].header
        h0.close()

        keys2 = [ float(header0['EXPSTART']), float(header0['EXPEND']) ]
        
        date_midexp = 0.5* ( float(keys2[0]) + float(keys2[1]) ) + 2400000.5 ##-- JD

        if not (date_midexp > minJD)*(date_midexp<maxJD):
            print k, date_midexp, (date_midexp > minJD)*(date_midexp<maxJD)
            continue
        
        dx, dy, dz = float(Xi(date_midexp)), float(Yi(date_midexp)), float(Zi(date_midexp))
        #print dx, dy, dz

        #d = date2jd( keys[3] )
        #dx.append( [ d, float(keys[0])*km2AU, float(keys[1])*km2AU, float(keys[2])*km2AU, k.split('_')[0] ] )

        prihdr = pyfits.Header()
        prihdr['MIDMJD']      = date_midexp - 2400000.5
        prihdr['MIDJD']      = date_midexp #- 2400000.5
        prihdr['POSTNSTX']    = dx
        prihdr['POSTNSTY']    = dy
        prihdr['POSTNSTZ']    = dz
        
        prihdr['COMMENT'] = 'Generated by update_spt.py on %s'%(time.ctime())
        prihdu = pyfits.PrimaryHDU(header=prihdr)

        prihdu.writeto('/net/frakir/raid/aparker/FullSurvey/%s_sdt.fits'%(k.split('_')[0]))

if __name__ == '__main__':

    new_spt()
