import numpy, commands, math, os, random, pylab
from scipy.stats import gaussian_kde as kde
from scipy.special import jv
from scipy.spatial import KDTree as kdtree
from numpy import sin, cos, arctan, sqrt, tan, pi
import sys
from scipy.stats import chi2
import datetime
from matplotlib.patches import Ellipse

#path2oorb = '/Users/AlexParker/Downloads/oorb-read-only/main/oorb'

### Use oorb to couple between Jean Marcs orbital files and
### fit_radec / predict

class OorbConf:
    def __init__( self, raw_conf='/Users/spica/scripts/raw_oorb.conf' ):

        handle = open(raw_conf, 'r')
        dat  = handle.readlines()
        handle.close()

        params = {}

        order = []
        for k in dat:
            if k[0] == '#' or k.strip() == '':
                #order.append( k )
                continue
            else:
                k = k.strip().split(':')
                params[ k[0].strip() ] = k[-1].strip()
                order.append(  k[0].strip()  )

        self.params = params
        self.order = order

    def nice( self, keyterm=None ):

        if not keyterm is None:
            for key in self.params.keys():
                if key.split('.')[0] == keyterm:
                    print '%s: %s'%( key, self.params[ key ] )
        else:
            for key in self.params.keys():
                print '%s: %s'%( key, self.params[ key ] )

    def write( self, path='./py_oorb.conf' ):

        handle = open(path, 'w')
        for key in self.order:
            #if key[0] == '#' or key.strip() == '':
            #    handle.write( key )
            #else:
            handle.write( '%s:    %s\n'%( key, self.params[key] ) )
        handle.write('\n')
        handle.close()



def repstr(s1, s2, i0):
    l = '%s%s%s' % (s1[0:i0], s2, s1[i0 + len(s2):])

    #for i in range(0, len(s2)):
    #	s1[i + i0] = s2[i]
    return l


class mpc_line:
    def __init__(self, JD=2456173.500000,
                 RA='00:00:00.000',
                 DEC='+00:00:00.00',
                 NAME='TMPMPC',
                 OBSCODE='500', obsdatI=''):

        JD2DJD = 2415020.0
        J2000 = 2451544.500000
        T0 = datetime.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        dt = datetime.timedelta(days=JD - J2000)

        T1 = T0 + dt
        D = T1.timetuple()

        self.obscode = OBSCODE
        self.name = NAME
        self.jd = JD
        self.ra = RA
        self.dec = DEC

        self.MODE = 'S'
        self.mode = 's'

        self.year = D.tm_year
        self.month = D.tm_mon
        self.day = D.tm_mday + D.tm_hour / 24.0 + D.tm_min / 1440.0 + D.tm_sec / 86400.0

        self.discovery_obs = ''
        self.mag = '25.0'
        self.fragment = ''
        self.progcode = ''

        self.hst_offset = numpy.asarray([0, 0, 0])

    def convert(self):

        line = ' '.ljust(80, ' ')

        ### PACKED DESIGNATION FROM NAME
        if len(self.name) > 15:
            self.name = self.name[0:15]

        #print len(line)

        line = repstr(line, self.name.ljust(15, ' '), 5)
        line = repstr(line, str(self.discovery_obs), 12)

        line = repstr(line, str(self.MODE), 14)

        if int(self.day) < 10:
            self.day = '0%s' % (self.day)
        else:
            self.day = '%s' % (self.day)

        if len(self.day) > 7:
            self.day = self.day[0:8]

        line = repstr(line, '%s %s %s' % ( str(self.year), str(self.month).rjust(2, '0'), str(self.day).ljust(8, '0')),
                      15)

        #line = repstr( line,str(self.day).ljust(12, '0'), 27)
        line = repstr(line, str(self.ra).ljust(12, ' '), 32)
        line = repstr(line, str(self.dec).ljust(12, ' '), 44)

        line = repstr(line, str(self.mag), 65)
        line = repstr(line, str(self.progcode).ljust(1, ' '), 13)

        line = repstr(line, str(self.obscode), 77)

        line2 = ' '.ljust(80, ' ')
        line2 = repstr(line2, self.name.ljust(15, ' '), 5)
        line2 = repstr(line2, str(self.discovery_obs), 12)

        line2 = repstr(line2, str(self.mode), 14)

        line2 = repstr(line2,
                       '%s %s %s' % ( str(self.year), str(self.month).rjust(2, '0'), str(self.day).ljust(8, '0')), 15)

        line2 = repstr(line2, '2', 32)

        signx = '+'
        if self.hst_offset[0] < 0:
            signx = '-'
        line2 = repstr(line2, signx, 34)

        line2 = repstr(line2, '%.8f' % ( abs(self.hst_offset[0])), 35)

        signy = '+'
        if self.hst_offset[1] < 0:
            signy = '-'
        line2 = repstr(line2, signy, 46)

        line2 = repstr(line2, '%.8f' % ( abs(self.hst_offset[1])), 47)

        signz = '+'
        if self.hst_offset[2] < 0:
            signz = '-'
        line2 = repstr(line2, signz, 58)

        line2 = repstr(line2, '%.8f' % ( abs(self.hst_offset[2])), 59)

        line2 = repstr(line2, str(self.obscode), 77)


        return line, line2

        ### line definitions
        ### 1-16 (A16) Packed designation(s)
        ### 17-18 (A2) Fragment designation
        ### 19   ('1') Indicates that this is the
        ###            first record for this observation.
        ###            This character will be 'A' for
        ###            the first record of a secondary comp.
        ### 20    (A1) = '*' if a discovery observation
        ###            = '+' if last record.
        ### 21    (A1) Mode of observation:
        ###            = C for CCD
        ###            = R for radar
        ###            = T for transit-circle / meridian-circ obs.
        ###            = M for visual micrometer observations
        ###            = S for satellite-based observations
        ###            = P for photographic observations
        ###            = O for offset observations
        ###            = E for occultation observations
        ###            = A for observations adj. to J2000 from B1950
        ###            = ' ' for unknown/unspecified
        ### 22-25   I4  Year of observation
        ### 26-27   I4  Month of observation
        ### 28-39 F12.9 Day of observation
        ### 40      A1 Time scale for time of obs (ASCII 32 = UTC)
        ###
        ### 41-52 A12  J2000 RA in HH MM SS.ddd
        ### 56-67 A12  J2000 DEC in +/-DD MM SS.dd
        ###   OR
        ### 41-52 F15.14 J2000 RA in radians
        ### 56-67 F16.14 J2000 DEC in radians
        ###
        ### 72-80  A9   Magnitude
        ###
        ### stuff for radar, offset...
        ###
        ### 82-84  A3 Up to three alphabetic notes
        ### 85     A1 Telescope identifier (1 = 1st telescope, 2=2nd...)
        ### 86-123 (blank on new submissions).
        ### 124    A1 MPC-assigned program code
        ### 129-132 A4 Obs code


def decdeg2dms(dd, hrs=False):
    Dsign = '+'
    if dd < 0:
        Dsign = '-'
        dd = abs(dd)

    if hrs:
        dd *= 24.0 / 360.0  ### Translate to HH MM SS.sss

    ### Otherwise just translate to DD MM SS.sss

    mntR, secR = divmod(dd * 3600, 60)
    degR, mntR = divmod(mntR, 60)

    #degR, mntR, secR = float(degR), float(mntR), float(secR)

    if hrs:
        return '%s:%s:%s' % (
            str('%d' % degR).rjust(2, '0'), str('%d' % mntR).rjust(2, '0'), str('%.3f' % (secR)).rjust(6, '0') )
    else:
        return '%s%s:%s:%s' % (
            Dsign, str('%d' % degR).rjust(2, '0'), str('%d' % mntR).rjust(2, '0'), str('%.2f' % (secR)).rjust(5, '0') )


def get_r(M, sma, ecc):
    ### very fast series solution of Kepler's equation to 5th order. OK for low ecc of HST's orbit.

    E = M + ( jv(1, 1.0 * ecc) * sin(1.0 * M) * 2.0 / float(1.0) +
              jv(2, 2.0 * ecc) * sin(2.0 * M) * 2.0 / float(2.0) +
              jv(3, 3.0 * ecc) * sin(3.0 * M) * 2.0 / float(3.0) +
              jv(4, 4.0 * ecc) * sin(4.0 * M) * 2.0 / float(4.0) +
              jv(5, 5.0 * ecc) * sin(5.0 * M) * 2.0 / float(5.0) )

    T = 2.0 * numpy.arctan2(numpy.sqrt(1.0 + ecc) * numpy.sin(0.5 * E), numpy.sqrt(1.0 - ecc) * numpy.cos(0.5 * E))
    R = sma * (1.0 - ecc ** 2) / ( 1.0 + ecc * numpy.cos(T) )

    return R, T


def tilt_Orbit(r, phi, inc, peri, OM):
    ### expects radians for all angular params

    return (r * cos(peri + phi) * cos(OM) - r * sin(peri + phi) * sin(OM) * cos(inc),
            r * cos(peri + phi) * sin(OM) + r * sin(peri + phi) * cos(OM) * cos(inc),
            r * sin(peri + phi) * sin(inc))


def approxHST(time=2456615.500000000):
    # 2456615.500000000 = A.D. 2013-Nov-19 00:00:00.0000 (CT)
    # EC= 4.989817886833338E-04 QR= 4.634738933719388E-05 IN= 2.844895649028797E+01
    # OM= 3.375029607622650E+02 W = 3.327960631993894E+02 Tp=  2456615.501612665132
    # N = 5.409462173747263E+03 MA= 3.512763492033158E+02 TA= 3.512676715655634E+02
    # A = 4.637052738589108E-05 AD= 4.639366543458827E-05 PR= 6.655005404180865E-02

    D2R = math.pi / 180.0

    sma = 4.637052738589108E-05
    ecc = 4.989817886833338E-04
    inc = 2.844895649028797E+01 * D2R

    MA = 3.512763492033158E+02 * D2R
    OM = 3.375029607622650E+02 * D2R
    om = 3.327960631993894E+02 * D2R

    TAU = 6.655005404180865E-02  ### Days

    dt = time - 2456615.500000000  ### Epoch for coords above.

    MA_now = math.atan2(math.sin(MA + dt * 2.0 * math.pi / TAU), math.cos(MA + dt * 2.0 * math.pi / TAU))

    if MA_now < 0.0:
        MA_now += 2.0 * math.pi

    r, phi = get_r(MA_now, sma, ecc)
    x, y, z = tilt_Orbit(r, phi, inc, om, OM)

    return numpy.asarray([x, y, z])


class astorb:
    def __init__(self, desig, epoch, orb):
        self._oorb = numpy.concatenate(( [desig], orb, [epoch],))


def ec2eq(x, y, z, HST_OFFSET=numpy.asarray([0.0, 0.0, 0.0]), T=0.0, noise=0.0):

    ep = (23.4392794444 - (46.836769 * T / 3600.0) - (0.0001831 * T ** 2 / 3600.0) + (0.0020034 * T ** 3 / 3600.0) - (
        0.576E-6 * T ** 4 / 3600.0) - (4.34E-8 * T ** 5 / 3600.0) ) * math.pi / 180.0

    M1 = numpy.matrix([[1.0, 0.0, 0.0],
                       [0.0, math.cos(ep), -math.sin(ep)],
                       [0.0, math.sin(ep), math.cos(ep)]])

    M2 = numpy.matrix([[x],
                       [y],
                       [z]])

    xyz_eq = numpy.ravel(numpy.dot(M1, M2))

    ### NOW ADD HST TOPOCENTRIC OFFSETS
    xyz_eqHST = xyz_eq - HST_OFFSET  ### <--- NOTE TO SELF: this SHOULD be a difference - vector offset to spacecraft.

    r = numpy.sum(xyz_eqHST ** 2) ** 0.5
    xyz_eqHST /= r

    lon = numpy.arctan2(xyz_eqHST[1], xyz_eqHST[0])

    if lon < 0:
        lon += 2.0 * math.pi

    lat = numpy.arcsin(xyz_eqHST[2])


    ### NOW ADD NOISE, IF REQUIRED
    if noise >= 1E-6:
        drv = abs(numpy.random.normal(0, noise * math.pi * math.sqrt(2) / ( 180.0 * 3600.0 )))
        dtv = numpy.random.uniform(0, 2.0 * math.pi)

        xrv = drv * numpy.cos(dtv)
        yrv = drv * numpy.sin(dtv)

        lon += yrv / numpy.cos(lat)
        lat += xrv

    return lon, lat


def jm2oorb(orb0, H=10.0):
    L = '------0001-----<>--------0002--------<>---------0003-------<>--------0004--------<>--------0005--------<>--------0006--------<>--------0007--------<>------0074-----<>----0038---<>-------0030------<>-------0031------<>-------0032------<>-------0033------<>-------0034------<>-------0035------<>---0036-<>---0037-<'.strip().split(
        '>')

    Lout = ''

    for i in range(0, len(L)):
        #print i
        if i < len(orb0):  ### Orbital elements, etc.

            try:
                Ci = '%.8f' % ( float(orb0[i]) )
            except:
                Ci = orb0[i]
            Lout += '%s' % ( str(Ci).ljust(len(L[i]) + 1, ' ') )
        elif i == len(orb0):  ### Translation type
            Lout += str('cartesian').ljust(len(L[i]) + 1, ' ')
        elif i == len(L) - 2:  ### H magnitude
            Lout += str(H).ljust(len(L[i]) + 1, ' ')
        elif i == len(L) - 1:  ### G slope param
            Lout += str('0.15').ljust(len(L[i]) + 1, ' ')
        else:  ### Unity placeholders for probability columns
            Lout += str('1.00').ljust(len(L[i]) + 1, ' ')

    return Lout


### Step 1: Propagate orbit to observation epoch
#oorb --task=propagation --epoch-mjd-tt=MJD --orb-in=INFILE [ --orb-out=OUTFILE ]
def prop2epoch(orbfile, epoch, path2oorb='oorb --conf=./py_oorb.conf'):
    #print 'Starting orbit propogation...'
    res = commands.getoutput('%s --task=propagation --epoch-mjd-utc=%s --orb-in=%s --orb-out=PROP2EPOCH_%s' % (path2oorb,
                                                                                                                epoch,
                                                                                                                orbfile,
                                                                                                                orbfile )).split('\n')

    os.system('cp PROP2EPOCH_%s last_prop.orb' % ( orbfile ))

    #print 'Propogation done.'

    return res


### Step 2: Compute ephemeris at observation epoch.
#oorb --task=ephemeris --obs-code=CODE [ --epoch-mjd-utc=MJD | --epoch-mjd-tt=MJD | --timespan=DT1 --step=DT2 ] --orb-in=INFILE
def oorb_ephem(orbfile, path2oorb='oorb --conf=./py_oorb.conf'):
    res = commands.getoutput('%s --task=ephemeris --obs-code=500 --orb-in=PROP2EPOCH_%s ' % (path2oorb,
                                                                                             orbfile )).split('\n')
    #print res
    keydict = {}
    KEYS = res[0][1:].strip().split()
    for i in range(0, len(KEYS)):
        keydict[KEYS[i]] = i

    results = []
    for k in res[1:]:
        k = k.strip().split()
        results.append(k)

    return keydict, results


def is_number(s):
    try:
        s2 = float(s)
        return True
    except ValueError:
        return False


def getastrom(orbfile, epoch_obs='56799.50000000', epoch0='56799.50000000'):

    ### from NH_cold and NH_hot files, JD epoch is 2456800.00000 

    FILE = open(orbfile, 'r')
    dat = FILE.readlines()
    FILE.close()

    orbv = []
    new_orbv = []

    counter = 0

    FILE = open('test_orb.orb', 'w')

    FILE.write(
        '# \n# \n# \n#-----0001-----<>--------0002--------<>---------0003-------<>--------0004--------<>--------0005--------<>--------0006--------<>--------0007--------<>------0074-----<>----0038---<>-------0030------<>-------0031------<>-------0032------<>-------0033------<>-------0034------<>-------0035------<>---0036-<>---0037-<\n')

    for k in dat:
        k = k.strip().split()

        counter += 1

        #if len(k) > 8:
        if not is_number(k[0]):
            epoch0 = k[7]

            orb = astorb('obj%s' % (counter), k[7], k[1:7])
        else:
            orb = astorb('obj%s' % (counter), epoch0, k[0:6])

        orbv.append([orb._oorb, k[6]])
        new_orb = jm2oorb(orbv[-1][0])
        FILE.write('%s\n' % (new_orb))

    FILE.close()


def syn_astrom(orbv, datev, noise):
    FILE = open('all_orbs.intermediate', 'w')

    for k in orbv:
        FILE.write(k)
    FILE.close()

    getastrom('all_orbs.intermediate')

    all_astrom = []
    for k in orbv:
        all_astrom.append([])

    for k in datev:

        err_output = prop2epoch('test_orb.orb', k)
        astrom_keys, astrom = oorb_ephem('test_orb.orb')

        for i in range(0, len(orbv)):
            dx = float(astrom[i][astrom_keys['HEclObj_X']]) - float(astrom[i][astrom_keys['HEclObsy_X']])
            dy = float(astrom[i][astrom_keys['HEclObj_Y']]) - float(astrom[i][astrom_keys['HEclObsy_Y']])
            dz = float(astrom[i][astrom_keys['HEclObj_Z']]) - float(astrom[i][astrom_keys['HEclObsy_Z']])

            HST_OFF = approxHST(time=k + 2400000.5)

            ra, dec = ec2eq(dx, dy, dz, HST_OFFSET=HST_OFF, T=-0.0009, noise=noise)

            ra_s = decdeg2dms(ra * 180.0 / math.pi, hrs=True)
            dec_s = decdeg2dms(dec * 180.0 / math.pi, hrs=False)

            all_astrom[i].append([ra * 180.0 / math.pi, dec * 180.0 / math.pi, '%.8f' % (HST_OFF[0]), '%.8f' % (HST_OFF[1]), '%.8f' % (HST_OFF[2])])

    return all_astrom



def do_main(datev):
    FILE = open('/Users/spica/Downloads/NH_encounters_2014-2015/nhfile.txt', 'r')
    dat = FILE.readlines()
    FILE.close()

    dataset = []
    for k in dat:
        k = k.strip().split()
        #if k[0] != '44.4000' or k[1] != '0.0412':
        #    continue
        
        #dataset.append(numpy.asarray(k, dtype='float'))
        #dataset.append(numpy.asarray(k, dtype='float'))
        dataset.append(numpy.asarray(k, dtype='float'))

    orbv0 = []
    for j in dataset:
        orbv0.append('%s\n' % (' '.join(list(numpy.asarray(j, dtype='str')))))

    astrom = syn_astrom(orbv0, datev, noise=0.0 )
    astrom = numpy.asarray(astrom)

    return astrom, orbv0


def opt_rates( astrom, dates, orbits ):

    dt = (dates[1] - dates[0])*( 86400.0 )
    dt_scale = 384.0 / dt
    #print dt, dt_scale
    sRA, sDEC = [],[]
    #print datev[1:] - datev[:-1]


    x,y,c = [],[],[]
    x2,y2,c2 = [],[],[]

    FILE = open('loc.dat', 'w')
    FILE.write( '#$%s\n'%( dates[0] +2400000.5) )
    
    for i in range(0, len(astrom)):
        RA, DEC = float(astrom[i][0][0]), float(astrom[i][0][1])
        #print RA, DEC
        FILE.write('%s %s\n'%( RA, DEC ) )
        
        color = float( orbits[i].split()[-1] )
        x.append( RA * math.pi / 180.0 )
        y.append( DEC * math.pi / 180.0 )
        c.append( color )
    FILE.close()

    for i in range(0, len(astrom)):
        RA, DEC = float(astrom[i][-1][0]), float(astrom[i][-1][1])
        #print RA, DEC
        color = float( orbits[i].split()[-1] )
        x2.append( RA * math.pi / 180.0 )
        y2.append( DEC * math.pi / 180.0 )
        c2.append( color )


    mx1 = numpy.median( x )
    mx2 = numpy.median( x2 )

    mex, mey = numpy.mean( x ), numpy.mean( y )

    
    my1 = numpy.median( y )
    my2 = numpy.median( y2 )


    #print 'Offset between median and mean:'
    #print (mx1- mex)*math.cos( mx1 ) * 180.0 * 3600.0 / math.pi
    #print (my1- mey) *180.0 * 3600.0 / math.pi

    #print '------------'
    
    dx1 = numpy.asarray( x - mx1 )*math.cos( my1 )
    dx2 = numpy.asarray( x2 - mx2 )*math.cos( my2 )

    dy1 = numpy.asarray( y - my1 )
    dy2 = numpy.asarray( y2 - my2 )


    c = []
    for i in range(0, len(astrom)):
        
        #RA0, DEC0 = float(astrom[i][0][0]), float(astrom[i][0][1])
        shiftRA, shiftDEC = [],[]
        for j in range(1, len(datev)):
            if abs( datev[j] - datev[j-1] ) > 2.0E-2:
                ### skip if longer than an orbit between
                continue
            RA, DEC = float(astrom[i][j][0]), float(astrom[i][j][1])
            RA0, DEC0 = float(astrom[i][j-1][0]), float(astrom[i][j-1][1])
            #print RA0, DEC0, RA, DEC
            dRA, dDEC = math.cos( DEC0 * math.pi / 180.0 ) * (RA - RA0), (DEC - DEC0)
            shiftRA.append( dRA )
            shiftDEC.append( dDEC )
        #print len(shiftRA), len(datev)
        sRA.append( shiftRA )
        sDEC.append( shiftDEC )

    sRA, sDEC = numpy.asarray( sRA ), numpy.asarray( sDEC )
    dv = []
    for i in range(0, len(sRA)):
        deltaRA  = ( sRA - sRA[i] )**2
        deltaDEC = ( sDEC - sDEC[i] )**2
         
        delta = numpy.percentile( numpy.sqrt( numpy.mean( deltaRA + deltaDEC, axis=1 )), 95 )
        #delta = numpy.mean( numpy.sqrt( numpy.mean( deltaRA + deltaDEC, axis=1 )))

        dv.append( delta )
        c.append( delta )


    
        
    pylab.ion()
    #pylab.scatter( (dx1-dx2)*180.0 *3600.0/ (math.pi),(dy1-dy2)*180.0 *3600.0/ (math.pi), c=c )
    r = numpy.sqrt( ( (dx1-dx2)*180.0 *3600.0/ (math.pi))**2 + ( (dy1-dy2)*180.0 *3600.0/ (math.pi))**2 )
    r2 = numpy.sqrt( ( (dx1)*180.0 *3600.0/ (math.pi))**2 + ( (dy1)*180.0*3600.0/ (math.pi))**2 )

    pylab.scatter( r, r2, c=c )
    pylab.scatter( [r[ numpy.argmin( r ) ]], [r2[ numpy.argmin( r ) ]], s=100 )
    pylab.draw()
    pylab.draw()

    print numpy.argmin( r )
    print numpy.asarray(orbits)[ numpy.argmin( r ) ]
    print dx1[ numpy.argmin( r ) ]*180.0 *3600.0/ (math.pi), dy1[ numpy.argmin( r ) ]*180.0 *3600.0/ (math.pi)
    
    pause = raw_input('...')
        
    pylab.clf()
    pylab.hist( dt_scale * numpy.asarray( dv ) * 3600.0 / 0.04, bins=30 )
    pylab.draw()
    pylab.draw()

    v = numpy.asarray(orbits)[ numpy.argsort( dv ) ][0:50]
    d, sma = [],[]
    for k in v:
        k = k.strip().split()
        d.append( float( k[-1] ) )
        sma.append( float(k[0] ) )

    print min(d), max(d)
    print min(sma), max(sma)
    
    #print dt_scale * min(dv) * 3600.0 / 0.04, numpy.argmin( dv )
    #print dv[912] * 3600.0 / 0.04, dv[958] * 3600.0 / 0.04
    pause = raw_input('...')
    return 0
        

    
if __name__ == '__main__':

    pylab.ion()

    OC = OorbConf()
    OC.params['sor.norb'] = 1500
    OC.params['reg.pdf'] = 'F'
    
    OC.write()


    ### April 15, May 15, June 15
    arc_length = 6.655005404180865E-02  ### days
    t0 = 2456839.500000 - 2400000.5

    if os.path.isfile('./DATES'):
        handle_dates = open('./DATES', 'r')
        dlines = handle_dates.readlines()
        handle_dates.close()

        t0 += float( dlines[0].strip())
        delta_t = numpy.asarray( dlines[1].strip().split(), dtype='float') * arc_length

        print 'DATES found:'
        print 't0 + %s'%( float( dlines[0].strip()) )
        print delta_t
    else:
        delta_t = [ 0 ] #, 3.0 * arc_length ]


    datev = []

    if os.path.isfile('./DATES'):
        for k in delta_t:
            datev = numpy.append(datev,
                                numpy.linspace(t0 + k - 0.5 * 0.61 * arc_length, t0 + k + 0.5 * 0.61 * arc_length, 5))
    else:
        datev = numpy.asarray([t0, t0])

    print datev +  2400000.5
    astrom, orbv = do_main(numpy.asarray(datev))

    best_orb_ind = opt_rates( astrom, datev, orbv )
        
