import numpy, sys, commands



def date2jd( hstdate ):
    d2 = hstdate.split(':')
    days = ( ( float( d2[0].split('.')[0]) - 2014.0 ) * 365.25 + float( d2[0].split('.')[1] ) + 
                float( d2[1] )/24.0 + float( d2[2] ) / 1440.0 + float( d2[3] ) / 86400.0 ) + 2456658.500000 - 1.0 ##-- normed to 2014, corrected to day 0

    return days #- 2400000.5


fnv = commands.getoutput('ls *.asc').split('\n')

ALLFILE = open('/Users/spica/Downloads/hubble_orbit_ephemeris_full_jd.asc', 'w')
for f in fnv:
    FILE = open(f, 'r')
    dat  = FILE.readlines()
    FILE.close()

    jdv = []
    for k in dat:
        k = k.strip().split()
        jd = date2jd( k[0] )

        print jd
        ALLFILE.write('%s %s\n'%( jd, ' '.join(k[1:]) ) )
        #pause = raw_input('...')

ALLFILE.close()
