import numpy, commands, math, os, random, pylab
from scipy.stats import gaussian_kde as kde
from scipy.special import jv
from scipy.spatial import KDTree as kdtree
from numpy import sin, cos, arctan, sqrt, tan, pi
import sys, time
from scipy.stats import chi2
import datetime
from matplotlib.patches import Ellipse, Circle
from scipy.optimize import fmin

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

class astorb:
    def __init__(self, desig, epoch, orb):
        self._oorb = numpy.concatenate(( [desig], orb, [epoch],))

def ec2eq(x, y, z, HST_OFFSET=numpy.asarray([0.0, 0.0, 0.0]), T=0.0, noise=0.0):

    ### no noise here.
    ep = (23.4392794444 - (46.836769 * T / 3600.0) - (0.0001831 * T ** 2 / 3600.0) + (0.0020034 * T ** 3 / 3600.0) -
          (0.576E-6 * T ** 4 / 3600.0) - (4.34E-8 * T ** 5 / 3600.0) ) * math.pi / 180.0

    #ep = 23.4392911111 * math.pi / 180.0

    M1 = numpy.matrix([[1.0, 0.0, 0.0],
                       [0.0, math.cos(ep), -math.sin(ep)],
                       [0.0, math.sin(ep),  math.cos(ep)]])

    M2 = numpy.matrix([[x],
                       [y],
                       [z]])

    xyz_eq = numpy.ravel(numpy.dot(M1, M2))

    ### NOW ADD HST TOPOCENTRIC OFFSETS
    xyz_eqHST = xyz_eq - HST_OFFSET  ### <--- NOTE TO SELF: this SHOULD be a difference - vector offset to spacecraft.

    r = numpy.sum( xyz_eqHST**2 ) ** 0.5
    xyz_eqHST /= r

    lon = numpy.arctan2(xyz_eqHST[1], xyz_eqHST[0])

    if lon < 0:
        lon += 2.0 * math.pi

    lat = numpy.arcsin(xyz_eqHST[2])

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
            #Lout += str('cartesian').ljust(len(L[i]) + 1, ' ')
            Lout += str('keplerian').ljust(len(L[i]) + 1, ' ')

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
    return res


### Step 2: Compute ephemeris at observation epoch.
#oorb --task=ephemeris --obs-code=CODE [ --epoch-mjd-utc=MJD | --epoch-mjd-tt=MJD | --timespan=DT1 --step=DT2 ] --orb-in=INFILE
def oorb_ephem(orbfile, path2oorb='oorb --conf=./py_oorb.conf'):
    res = commands.getoutput('%s --task=ephemeris --obs-code=500 --orb-in=PROP2EPOCH_%s ' % (path2oorb,
                                                                                             orbfile )).split('\n')
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

        err_output = prop2epoch('test_orb.orb', k[0])
        astrom_keys, astrom = oorb_ephem('test_orb.orb')

        for i in range(0, len(orbv)):
            dx = float(astrom[i][astrom_keys['HEclObj_X']]) - float(astrom[i][astrom_keys['HEclObsy_X']])
            dy = float(astrom[i][astrom_keys['HEclObj_Y']]) - float(astrom[i][astrom_keys['HEclObsy_Y']])
            dz = float(astrom[i][astrom_keys['HEclObj_Z']]) - float(astrom[i][astrom_keys['HEclObsy_Z']])

            HST_OFF = numpy.asarray( [k[1], k[2], k[3]], dtype='float' )

            ra, dec   = ec2eq(dx, dy, dz, HST_OFFSET=HST_OFF,                      T=-0.0009, noise=0.0)
            ra2, dec2 = ec2eq(dx, dy, dz, HST_OFFSET=numpy.asarray([0.0,0.0,0.0]), T=-0.0009, noise=0.0)

            #print k[-1], 3600.0*(ra-ra2)*180.0/math.pi, 3600.0*(dec-dec2)*180.0/math.pi
            #pause = raw_input('...')
            
            all_astrom[i].append([ra * 180.0 / math.pi, dec * 180.0 / math.pi,
                                  '%.8f' % (HST_OFF[0]), '%.8f' % (HST_OFF[1]), '%.8f' % (HST_OFF[2]),
                                  ra2 * 180.0 / math.pi, dec2 * 180.0 / math.pi])

    return all_astrom

def do_main(datev):
    FILE = open('/Users/spica/scripts/git_projects/skyrate/hst_field_orbits.txt', 'r')
    dat = FILE.readlines()
    FILE.close()

    dataset = []
    for k in dat:
        k = k.strip().split()
        dataset.append(numpy.asarray(k, dtype='float'))

    orbv0 = []
    for j in dataset:
        orbv0.append('%s\n' % (' '.join(list(numpy.asarray(j, dtype='str')))))

    astrom = syn_astrom(orbv0, datev, noise=0.0 )
    astrom = numpy.asarray(astrom)

    return astrom, orbv0

def grid( theta, aRA, aDEC, epsilon=0.04, full=False, count_thresh=3 ):

    dx = math.sqrt(3) * epsilon
    dy = 1.5 * epsilon

    sRA  = math.cos(theta)*aRA - math.sin(theta)*aDEC
    sDEC = math.sin(theta)*aRA + math.cos(theta)*aDEC
        
    minD, maxD = min(sDEC) - 5*epsilon, max(sDEC) + 5*epsilon
    minR, maxR = min(sRA)  - 5*epsilon, max(sRA)  + 5*epsilon
        
    xg1 = numpy.unique( numpy.append( numpy.arange( 0, minR, -dx), numpy.arange(0, maxR, dx ) ) )
    xg2 = xg1 + 0.5 * dx
    
    yg1 = numpy.unique( numpy.append( numpy.arange( 0, minD, -2.0*dy), numpy.arange(0, maxD, 2.0*dy ) ) )
    yg2 = yg1 + dy

    X1, Y1 = numpy.meshgrid( xg1, yg1 )
    X2, Y2 = numpy.meshgrid( xg2, yg2 )

    X = numpy.append( X1.ravel(), X2.ravel() )
    Y = numpy.append( Y1.ravel(), Y2.ravel() )

    count = 0
    Xk, Yk, inds = [],[],[]
    for i in range(0, len(X)):
        r = (X[i]-sRA)**2 + (Y[i]-sDEC)**2
        ok = numpy.sum( r <= epsilon**2)
        if ok >= count_thresh:
            inds.append( numpy.argmin(r) )
            this_ind = numpy.argmin(r)
            ### move to best discrete orbit, test again:
            r2 = (sRA[this_ind]-sRA)**2 + (sDEC[this_ind]-sDEC)**2
            if numpy.sum( r2 <= epsilon**2 ) < count_thresh:
                continue
            
            count += 1

            Xk.append( X[i]*math.cos(-theta) - Y[i]*math.sin(-theta) )
            Yk.append( X[i]*math.sin(-theta) + Y[i]*math.cos(-theta) )
    if full:
        return count, Xk, Yk, numpy.sort( inds )
    else:
        return count  

def extract_offsets(astrom_line):

        ### find closest to median time
        RA0, DEC0 = float(astrom_line[2][0]), float(astrom_line[2][1])
        RA1, DEC1 = float(astrom_line[7][0]), float(astrom_line[7][1])

        cross_deltaRA, cross_deltaDEC = 3600.0 * math.cos( DEC0 * math.pi / 180.0 ) * (RA1 - RA0), 3600.0 * (DEC1 - DEC0)
        
        all_RA, all_DEC = [],[]
        shiftRA, shiftDEC = [],[]
        
        for i in range(0, len(astrom_line)):
            RA, DEC = float(astrom_line[i][0]), float(astrom_line[i][1])
            GRA, GDEC = float(astrom_line[i][-2]), float(astrom_line[i][-1])

            all_RA.append( [RA,GRA] )
            all_DEC.append( [DEC,GDEC] )
            
            if i <= 4:
                dRA, dDEC = 3600.0 * math.cos( DEC0 * math.pi / 180.0 ) * (RA - RA0), 3600.0 * (DEC - DEC0) ### delta-arcsec
            else:
                 dRA, dDEC = 3600.0 * math.cos( DEC1 * math.pi / 180.0 ) * (RA - RA1), 3600.0 * (DEC - DEC1) ### delta-arcsec

            shiftRA.append( dRA )
            shiftDEC.append( dDEC )

        return numpy.asarray( shiftRA ), numpy.asarray( shiftDEC ), numpy.asarray( all_RA ), numpy.asarray( all_DEC ), cross_deltaRA, cross_deltaDEC

def max_delta( sra1, sra2, sdec1, sdec2 ):
    dr = sra1-sra2
    dd = sdec1-sdec2
    delta = []
    for i in range(0, len(dr)-1):
        for j in range(i+1, len(dr)):
            delta.append( (dr[i] - dr[j])**2 + (dd[i] - dd[j])**2 )
            
    return max( delta )**0.5
            
def allmax_delta( sra, sdec, eps, N=300 ):
    all_count = {}

    dv1_frac, dv2_frac = [],[]
    for i in range(0, N ): 
        count = []

        dv1, dv2 = [],[]
        for j in range(0, len(sra)):
            d1 = max_delta( sra[i][0:5], sra[j][0:5], sdec[i][0:5], sdec[j][0:5] )
            d2 = max_delta( sra[i][5:],  sra[j][5:],  sdec[i][5:],  sdec[j][5:] )
            if d1 <= eps and d2 <= eps:
                dv1.append(d1)
                dv2.append(d2)
                count.append( j )

        dv1_frac.append( numpy.percentile( dv1, 95 ) / eps )
        dv2_frac.append( numpy.percentile( dv2, 95 ) / eps )

    s1, s2 = numpy.median( dv1_frac ), numpy.median( dv2_frac )

    
    print 'Percentiles: Median %.2f   %.2f'%( s1, s2 )
    print '             5th:   %.2f   %.3f'%( numpy.percentile( dv1_frac, 5 ), numpy.percentile( dv2_frac, 5 ) )
    print '            95th:   %.2f   %.3f'%( numpy.percentile( dv1_frac, 95), numpy.percentile( dv2_frac, 95) )
    
    if s1 > s2:
        print 'Gridding on Visit 1'
        return 0, s1
    else:
        print 'Gridding on Visit 2'
        return 1, s2
        
def opt_rates( astrom, datev, orbv, do_refined, orig_path ):


    #if os.path.isfile('/Users/spica/NHA/HST/demo_shifts/%s_all.tar'%(  datev[0][-1][0:6] ) ):
    #    print 'Seems like tarball already exists; will not regenerate.'
    #    return None

    root_filename = [] 
    for k in datev:
        root_filename.append( k[-1][0:6] )
        print k[-1][0:6], k[0]

    dates =  numpy.asarray( numpy.asarray(datev).T[0], dtype='float')
    
    deltav = []
    for k in orbv:
        k = k.strip().split()
        deltav.append( float(k[-1]) <= 0.130 )
    deltav = numpy.asarray( deltav )
    print 'MEAN ENCOUNTER PROB: %.3f'%( numpy.mean( deltav ) )

    
    print
    print 'Times:'
    print 'Visit 1 length: %.2f hours'%( (dates[4]-dates[0])*24.0 )
    print 'Visit 2 length: %.2f hours'%( (dates[9]-dates[5])*24.0 )
    print 'Mid-time gap:   %.2f hours'%( (dates[7]-dates[2])*24.0 )
    print

    root_filename = numpy.asarray( root_filename )
    root_filename =  numpy.unique( root_filename )

    FILE = open('%s/%s_%s_all/%s_%s_all.orbits'%(orig_path, root_filename[0], root_filename[1], root_filename[0], root_filename[1] ), 'r')
    orig_orbits = FILE.readlines()
    FILE.close()

    keep = []
    for k in orig_orbits:
        
        k2 = k.strip().split()

        orb_index = k2[0]
        orb_string = ' '.join(k2[1:])
        for j in range(0, len(orbv)):
            if orbv[j].strip() == orb_string:
                print 'Match orbit index %s with %s'%( orb_index, j )
                keep.append( j )
                break

    #sys.exit(-1)
    #return None

    dt = abs( dates - numpy.mean(dates) )
    center_ind = numpy.argmin( dt )
    
    dRA0, dDEC0, all_RA0, all_DEC0, crossRA0, crossDEC0 = extract_offsets(astrom[0])

    root_dir_path = '%s_%s_all'%( root_filename[0], root_filename[1] )
    root_dir_path0 = '%s_shifts'%( root_filename[0])
    root_dir_path1 = '%s_shifts'%( root_filename[1])
    
    if not os.path.isdir(root_dir_path):
        os.system('mkdir %s'%( root_dir_path ))

    if not os.path.isdir(root_dir_path0):
        os.system('mkdir %s'%( root_dir_path0 ))

    if not os.path.isdir(root_dir_path1):
        os.system('mkdir %s'%( root_dir_path1 ))
            
    FILE = open('./%s/%s_celestial.shifts'%(root_dir_path, root_filename[0]), 'w' )
    for i in range(0, 5):
        FILE.write( '%s %s %.4f %.4f\n'%( datev[i][-1], float(datev[i][0])+2400000.5, dRA0[i], dDEC0[i] ) )
    FILE.close()
    
    FILE = open('./%s/%s_celestial.shifts'%(root_dir_path, root_filename[1]), 'w' )
    for i in range(5, 10):
        FILE.write( '%s %s %.4f %.4f\n'%( datev[i][-1], float(datev[i][0])+2400000.5, dRA0[i], dDEC0[i] ) )
    FILE.close()    
        
    ### first line is the guiding orbit
    
    shiftsRA, shiftsDEC = [],[]
    cdRA, cdDEC = [],[]
    cgRA, cgDEC = [],[]
    shearRA, shearDEC = [],[]
    for i in range(0, len(astrom)):
        dRA, dDEC, all_RA, all_DEC, crossRA, crossDEC = extract_offsets(astrom[i])
        cdRA.append( all_RA )
        cdDEC.append( all_DEC )
        shiftsRA.append( dRA - dRA0 )
        shiftsDEC.append( dDEC - dDEC0 ) ## all arcsecs, normalized to guiding orbit
        shearRA.append( crossRA )
        shearDEC.append( crossDEC )
        #print shearRA[-1], shearDEC[-1]

        #pause = raw_input('...')
        
    shiftsRA, shiftsDEC = numpy.asarray(shiftsRA), numpy.asarray(shiftsDEC)
    shearRA, shearDEC   = numpy.asarray( shearRA ), numpy.asarray( shearDEC )

    which_orbit, scale_term = allmax_delta( shiftsRA, shiftsDEC, 0.04 )

    if which_orbit == 0:    
        sRA = shiftsRA[:,4] - shiftsRA[:,0]
        sDEC = shiftsDEC[:,4] - shiftsDEC[:,0]
    else:
        sRA = shiftsRA[:,9] - shiftsRA[:,5]
        sDEC = shiftsDEC[:,9] - shiftsDEC[:,5]

     

    '''
    print scale_term
    #thetav = numpy.arange(0,math.pi/2.0, math.pi / 180.0 )
    #NV = []
    
    #for theta in thetav:
    #    NV.append( grid( theta, sRA, sDEC, epsilon=scale_term*0.04 ) )
                   
    #best_theta = thetav[ numpy.argmin( NV ) ]
    #print min(NV), best_theta

    #N, Xk, Yk, keep = grid( best_theta, sRA, sDEC, epsilon=scale_term*0.04, full=True )

    #print N, len(keep), keep

    pylab.ion()
    fig = pylab.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, aspect='equal')
    ax.scatter(sRA, sDEC, c='k', zorder=-1)
    ax.scatter(sRA[keep], sDEC[keep], c='r', zorder=2 )
    
    for i in keep:
        ax.add_artist(Circle( xy=(sRA[i], sDEC[i]), radius=0.04, facecolor='none', edgecolor='blue', zorder=0  ))
        ax.add_artist(Circle( xy=(sRA[i], sDEC[i]), radius=scale_term*0.04, facecolor='none', edgecolor='cyan', zorder=1  ))

    ax.grid()
    ax.set_xlabel(r'max d$\alpha$ (arcsec)')
    ax.set_ylabel(r'max d$\delta$ (arcsec)')

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    
    pylab.savefig('./%s/%s_%s_NonlinearHexGrid.png'%(root_dir_path, root_filename[0], root_filename[1]), dpi=300) 
    pylab.draw()
    '''
    
    count = 0

    
    BIGFILE0 = open('./%s/%s_all.shifts'%(root_dir_path,root_filename[0]), 'w')
    BIGFILE1 = open('./%s/%s_all.shifts'%( root_dir_path, root_filename[1]), 'w')

    PLOTFILE = open('./%s/%s_%s_all.shear'%( root_dir_path, root_filename[0],  root_filename[1] ), 'w')

    ORBFILE  = open('./%s/%s_%s_all.orbits'%( root_dir_path, root_filename[0],  root_filename[1] ), 'w')
    
    for k in keep:
        #print k

        PLOTFILE.write('%s %.4f %.4f %.4f %.4f\n'%( str(count).rjust(3, '0'), sRA[k], sDEC[k], shearRA[k], shearDEC[k] ) )
    
        ORBFILE.write( '%s  %s'%( str(count).rjust(3, '0'), orbv[k] ) )
        #FILE = open('%s_%s_%s.shear'%(root_filename[0], root_filename[1], str(count).rjust(3, '0')), 'w')
        #FILE.write( '%s %s %.4f %.4f\n'%( datev[2][-1], datev[7][-1], shearRA[k], shearDEC[k] ) )
        #FILE.close()

        
        FILE = open('./%s/%s_%s.shifts'%( root_dir_path, root_filename[0], str(count).rjust(3, '0') ), 'w')
        for i in range(0, 5):
                FILE.write( '%s %.4f %.4f %.6f %.6f %.6f %.6f\n'%(    datev[i][-1],
                                                    shiftsRA[int(k)][i],
                                                    shiftsDEC[int(k)][i],
                                                    cdRA[int(k)][i][0],
                                                    cdDEC[int(k)][i][0],
                                                    cdRA[int(k)][i][1],
                                                    cdDEC[int(k)][i][1] ) )

                BIGFILE0.write( '%s %s %.4f %.4f %.6f %.6f %.6f %.6f\n'%( str(count).rjust(3, '0'), datev[i][-1],
                                                                shiftsRA[int(k)][i],
                                                                shiftsDEC[int(k)][i],
                                                                cdRA[int(k)][i][0],
                                                                cdDEC[int(k)][i][0],
                                                                cdRA[int(k)][i][1],
                                                                cdDEC[int(k)][i][1] ) )
        FILE.close()
        
        FILE = open('./%s/%s_%s.shifts'%( root_dir_path, root_filename[1], str(count).rjust(3, '0') ), 'w')
        for i in range(5,10):
                FILE.write( '%s %.4f %.4f %.6f %.6f %.6f %.6f\n'%(datev[i][-1],
                                                        shiftsRA[int(k)][i],
                                                        shiftsDEC[int(k)][i],
                                                        cdRA[int(k)][i][0],
                                                        cdDEC[int(k)][i][0],
                                                        cdRA[int(k)][i][1],
                                                        cdDEC[int(k)][i][1] ) )
                
                BIGFILE1.write( '%s %s %.4f %.4f %.6f %.6f %.6f %.6f\n'%( str(count).rjust(3, '0'), datev[i][-1],
                                                                shiftsRA[int(k)][i],
                                                                shiftsDEC[int(k)][i],
                                                                cdRA[int(k)][i][0],
                                                                cdDEC[int(k)][i][0],
                                                                cdRA[int(k)][i][1],
                                                                cdDEC[int(k)][i][1] ) )
        FILE.close()
        count += 1
        
    BIGFILE1.close()
    BIGFILE0.close()
    PLOTFILE.close()
    ORBFILE.close()

    os.system('cp %s/%s_*.shifts %s/'%( root_dir_path, root_filename[0], root_dir_path0))
    os.system('cp %s/%s_*.shifts %s/'%( root_dir_path, root_filename[1], root_dir_path1))

    ### remove the '_all' files from these paths
    os.system('rm %s/*_all.shifts'%(root_dir_path0))
    os.system('rm %s/*_all.shifts'%(root_dir_path1))
        
    os.system('tar -c %s > %s.tar'%( root_dir_path,  root_dir_path ) )
    os.system('tar -c %s > %s.tar'%( root_dir_path0, root_dir_path0 ) )
    os.system('tar -c %s > %s.tar'%( root_dir_path1, root_dir_path1 ) )


    #if not do_refined:
    #    os.system('cp *.tar /Users/spica/NHA/HST/demo_shifts/')
    #    #os.system('cp *.tar /Users/spica/NHA/HST/demo_shifts/hstshifts/')
    #    #os.system('scp *.tar aparker@suhuy:/net/suhuy/raid1/buie/hst/13633/shifts/')
    #else:
    os.system('cp *.tar /Users/spica/NHA/HST/hstfinalshifts/')
    #    #os.system('cp *.tar /Users/spica/NHA/HST/demo_shifts/hstshifts/')
    #os.system('scp *.tar aparker@suhuy:/net/suhuy/raid1/buie/hst/13633/shifts_final/')

        
        
    return None
        
    
def date2jd( hstdate ):
    d2 = hstdate.split(':')
    days = ( ( float( d2[0].split('.')[0]) - 2014.0 ) * 365.25 + float( d2[0].split('.')[1] ) + 
                float( d2[1] )/24.0 + float( d2[2] ) / 1440.0 + float( d2[3] ) / 86400.0 ) + 2456658.500000 - 1.0 ##-- normed to 2014, corrected to day 0

    return days - 2400000.5
        
def get_spt():
    
    fn = commands.getoutput('ls *_spt.fits').split('\n')
    fn2 = commands.getoutput('ls *_sdt.fits').split('\n')

    if len(fn2) >= len(fn):
        do_refined = True
    else:
        do_refined = False
        
    dx = []
    km2AU = 1.0 / 1.49598e8

    if not do_refined:
        for k in fn:
            keys = commands.getoutput('gethead %s POSTNSTX POSTNSTY POSTNSTZ PSTRTIME'%(k)).split()
            d = date2jd( keys[3] )
            dx.append( [ d, float(keys[0])*km2AU, float(keys[1])*km2AU, float(keys[2])*km2AU, k.split('_')[0] ] )
    else:
        for k in fn2:
            keys = commands.getoutput('gethead %s POSTNSTX POSTNSTY POSTNSTZ MIDMJD'%(k)).split()
            dx.append( [  float(keys[3]), float(keys[0])*km2AU, float(keys[1])*km2AU, float(keys[2])*km2AU, k.split('_')[0] ] )        

    return numpy.asarray(dx), do_refined
        
        
if __name__ == '__main__':

    #get list of directories
    dirlist = commands.getoutput('ls -d */').split('\n')

    for d in dirlist:
        
        os.chdir(d)
        print d
        
        datev, do_refined = get_spt()

        if os.path.isfile('/Users/spica/NHA/HST/hstfinalshifts/%s_shifts.tar'%(  datev[0][-1][0:6] ) ): # and not do_refined:
            print 'Seems like tarball already exists; will not regenerate.'
            os.chdir('../')
            continue
        else:
            print 'No tarball found'
            
        if len(datev) != 10:
            print 'Fewer than 10 _spt files found in dir %s; skipping.'%( d )
            os.chdir('../')
            continue
        
        OC = OorbConf()
        OC.params['sor.norb'] = 1500
        OC.params['reg.pdf'] = 'F'
        OC.params['relativity'] = 'F'
    
        OC.write()

        astrom, orbv = do_main(datev)
        best_orb_ind = opt_rates( astrom, datev, orbv, do_refined, '%s/%s'%(sys.argv[1], d) )
        os.chdir('../')
