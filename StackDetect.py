from pyxtractor import pyx
import pylab, numpy, commands, os, sys, math
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
import pyfits
from matplotlib.patches import Circle
#s = 'getfits -o tmp_%s %s %s-%s %s-%s'%( tmp_id, im, x-30, x+30, y-30, y+30)
import warnings
warnings.filterwarnings('ignore')

def corners( x1, y1, x2, y2, delta=60):

    max_y = 4458
    max_x = 4170
    pos_deltax = min( (delta, int(max_x-x1), int(max_x-x2), ) )
    neg_deltax = min( (delta, int(x1), int(x2), ))
    pos_deltay = min( (delta, int(max_y-y1), int(max_y-y2), ) )
    neg_deltay = min( (delta, int(y1), int(y2), ))

    return pos_deltax, neg_deltax, pos_deltay, neg_deltay
    
def topng(im1, x1, y1, im2, x2, y2, tmp_id1, tmp_id2, ax1, ax2, ax3, ax4):

    os.system('rm tmp0.fits tmp1.fits tmp2.fits tmp3.fits')
    px, nx, py, ny = corners( x1, y1, x2, y2, delta=60)
    #print px, n
    s = 'getfits -o tmp0.fits %s %s-%s %s-%s'%( im1, int(x1-nx), int(x1+px), int(y1-ny), int(y1+py))
    a = commands.getoutput( s )
    print a
    print s
    h = pyfits.open('tmp0.fits')
    d1 = h[0].data
    h.close()

    s = 'getfits -o tmp1.fits %s %s-%s %s-%s'%( im2, int(x2-nx), int(x2+px), int(y2-ny), int(y2+py))
    a = commands.getoutput( s )
    h = pyfits.open('tmp1.fits')
    d2 = h[0].data
    h.close()    

    px, nx, py, ny = corners( x1, y1, x2, y2, delta=200)

    s = 'getfits -o tmp2.fits %s %s-%s %s-%s'%( im1, int(x1-nx), int(x1+px), int(y1-ny), int(y1+py))
    a = commands.getoutput( s )
    h = pyfits.open('tmp2.fits')
    d3 = h[0].data
    h.close()

    s = 'getfits -o tmp3.fits %s %s-%s %s-%s'%( im2, int(x2-nx), int(x2+px), int(y2-ny), int(y2+py))
    a = commands.getoutput( s )
    h = pyfits.open('tmp3.fits')
    d4 = h[0].data
    h.close()   
    
    vals = numpy.append( d1.ravel(), d2.ravel() )

    L = numpy.percentile( vals, 14 )
    U = numpy.percentile( vals, 98 )
    M = 0.5*(U + L)
    W = (U-L)
    #sdev0 = 0.5*( numpy.std(d1[ok1]) + numpy.std(d2[ok2]))

    usig, wsig = 2.0, 0.33


    ax1.cla()
    ax2.cla()

    ax3.cla()
    ax4.cla()
        
    ax1.imshow( numpy.tanh((d1-M)/W), cmap='gist_heat', interpolation='nearest', vmin=-1, vmax=1 )
    ax2.imshow( numpy.tanh((d2-M)/W), cmap='gist_heat', interpolation='nearest', vmin=-1, vmax=1 )
    ax3.imshow( numpy.tanh((d3-M)/W), cmap='gist_heat', interpolation='nearest', vmin=-1, vmax=1 )
    ax4.imshow( numpy.tanh((d4-M)/W), cmap='gist_heat', interpolation='nearest', vmin=-1, vmax=1 )


    ax3.add_artist( Circle(xy=(nx,ny), radius=20, facecolor='none', edgecolor='k' ) )
    ax4.add_artist( Circle(xy=(nx,ny), radius=20, facecolor='none', edgecolor='k' ) )
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
            
    pylab.draw()
    #pause = raw_input('...')
    return True #'1' in pause.strip().lower()
    
def get_data( ims_in, exten=0 ):
    ##-- Uses Pyxtractor (source extractor plugin) to extract high-sig
    ##-- sources from two images.
    
    T = pyx()
    T.params = ['XWIN_IMAGE', 'YWIN_IMAGE', 'XWIN_WORLD', 'YWIN_WORLD', 'MAG_APER', 'FLAGS', 'FWHM_IMAGE', 'FLUX_APER', 'FLUXERR_APER']

    if len(sys.argv) >= 5:
        print 'DETECT THRESH: ', sys.argv[4]
        T.options[ 'DETECT_THRESH' ] = sys.argv[4]
    else:
        T.options[ 'DETECT_THRESH' ] = 1.5
    T.options[ 'PHOT_APERTURES' ] = 2.0
    T.options[ 'CLEAN' ] = 'Y'
    #T.options[ 'FILTER_NAME' ] = '/Users/spica/Dropbox/scripts/proc_files/gauss_50_9x9.conv'
    T.options[ 'FILTER_NAME' ] = '/Users/spica/Dropbox/scripts/proc_files/gauss_30_5x5.conv'

    T.options[ 'FILTER' ] = 'Y'
    T.options[ 'DETECT_MINAREA' ] = 1
    T.options[ 'MAG_ZEROPOINT' ] = 32.2 - 6.4

    T.getcat( ims_in )
    T.cleanup()

    return T.catalog


def get_r( kept_x, kept_y, x, y ):
    r =  ( ( numpy.asarray( kept_x ) - x )**2 + ( numpy.asarray( kept_y ) - y )**2 )**0.5
    return min(r), numpy.argmin(r)
    
'''
def group_by_xy(x10,y10,x20,y20,image1,image2,flux10,flux20,eps):

    kept_x1, kept_y1 = x10[0],y10[0]
    kept_x2, kept_y2 = x20[0],y20[0]
    keep_flux1, keep_flux2 = [],[]
    keep_ims1, keep_ims2 = [],[]

    for i in range(0, len(x10[0])):
        keep_flux1.append( [ x10[0][i] ] )
        keep_flux2.append( [ x20[0][i] ] )
        keep_ims1.append( [image1[0]] )
        keep_ims2.append( [image2[0]] )
 
                
    for j in range(1, len(x10)):
        x1,y1,x2,y2,flux1,flux2=(x10[j],
                                    y10[j],
                                    x20[j],
                                    y20[j],
                                    flux10[j],
                                    flux20[j], )
        
        for i in range(0, len(x1)):

            mr1, ir1 = get_r( kept_x1, kept_y1, x1[i], y1[i] )
            mr2, ir2 = get_r( kept_x2, kept_y2, x2[i], y2[i] )

            if mr1 < eps and mr2 < eps and ir1 == ir2:
                keep_ims1[ir1].append( image1[j] )
                keep_ims2[ir2].append( image2[j] )
                keep_flux1[ir1].append( flux1[i] )
                keep_flux2[ir2].append( flux2[i] )
            else:
                kept_x1.append( x1[i] )
                kept_x2.append( x2[i] )
                
                kept_y1.append( y1[i] )
                kept_y2.append( y2[i] )
                keep_ims1.append( [image1[j]] )
                keep_ims2.append( [image2[j]] )
                keep_flux1.append( [flux1[i]] )
                keep_flux2.append( [flux2[i]] )

    for i in range(0, len(kept_x1)):
        Fi = numpy.argmax( numpy.asarray(  numpy.asarray(keep_flux1[i]) + numpy.asarray(keep_flux2[i]) ) )

        print
        print i, len(keep_ims1[i])
        print keep_ims1[i][Fi], kept_x1[i], kept_y1[i]
        print keep_ims2[i][Fi], kept_x2[i], kept_y2[i]
'''

def filter_obj_info( OBJ_INFO ):
    #OBJ_info.append( [ X1[i], Y1[i],
    #                   cRA10[i], cDEC10[i],
    #                   MAG1[i],
    #                   X2[j], Y2[j],
    #                   cRA20[j], cDEC20[j],
    #                   MAG2[j], SN1, SN2, NAME ] )

    ### filter on X1, X2, sn
    orig_obj = OBJ_INFO[:]
    OBJ_INFO = numpy.asarray( numpy.asarray( OBJ_INFO ).T[0:-1], dtype='float')

    print OBJ_INFO[10]
    print
    print OBJ_INFO[11]
    
    snv = OBJ_INFO[10]**2 + OBJ_INFO[11]**2

    
    X = OBJ_INFO[0]
    Y = OBJ_INFO[1]
    tree = KDTree( numpy.asarray( [X,Y] ).T )
    keep = []
    for i in range(0, len(X)):
        match_inds = tree.query_ball_point( [X[i],Y[i]], 3.0 )
        if numpy.all( snv[i] >= snv[ match_inds ] ):
            keep.append( orig_obj[i] )

    print ' Original length: %d'%( len(orig_obj) )
    print 'Collapsed length: %d'%( len(keep) )
    
    return keep
    
        
def match_radec( cat1, cat2, offset_ra, offset_dec, dt_ratio, count, imname1, imname2, ax1, ax2 ):

    '''
    FILE = open('xy19.reg', 'r')
    dat  = FILE.readlines()
    FILE.close()

    FILE = open('plant19.txt', 'r')
    dat2  = FILE.readlines()
    FILE.close()
    
    RA_plant, DEC_plant, MAG_plant = [],[],[]
    for k in dat:
        k = k.strip().split()
        RA_plant.append( float(k[0]) )
        DEC_plant.append( float(k[1]) )
    for k in dat2:
        k = k.strip().split()
        MAG_plant.append( float(k[6]) )
        
        #print RA_plant[-1], DEC_plant[-1]
        
    RA_plant, DEC_plant = numpy.asarray( RA_plant ), numpy.asarray( DEC_plant )
    MAG_plant = numpy.asarray( MAG_plant )
    '''
    cRA10, cDEC10 = numpy.asarray(cat1[ 'XWIN_WORLD' ][:]), numpy.asarray(cat1['YWIN_WORLD'][:])
    cRA20, cDEC20 = numpy.asarray(cat2[ 'XWIN_WORLD' ][:]), numpy.asarray(cat2['YWIN_WORLD'][:])

        
    RA1, DEC1 = 1.0*cat1[ 'XWIN_WORLD' ][:], 1.0*cat1['YWIN_WORLD'][:]
    RA2, DEC2 = 1.0*cat2[ 'XWIN_WORLD' ][:], 1.0*cat2['YWIN_WORLD'][:]

    
    #print RA1
    #print DEC1
    
    X1, Y1, X2, Y2 = cat1[ 'XWIN_IMAGE' ], cat1['YWIN_IMAGE'], cat2[ 'XWIN_IMAGE' ], cat2['YWIN_IMAGE']
    
    MAG1, MAG2 = cat1[ 'MAG_APER' ], cat2[ 'MAG_APER' ]
    SN1, SN2 = cat1[ 'FLUX_APER' ] / cat1[ 'FLUXERR_APER' ], cat2[ 'FLUX_APER' ] / cat2[ 'FLUXERR_APER' ]
    
    RA1 *= 3600.0 / numpy.cos( DEC1 * math.pi / 180.0 )
    RA2 *= 3600.0 / numpy.cos( DEC2 * math.pi / 180.0 )

    #RA_plant *= 3600.0 / numpy.cos( DEC_plant * math.pi / 180.0 )

    RA1 += offset_ra
    #RA_plant += offset_ra
    
    DEC1 *= 3600.0
    DEC2 *= 3600.0
    #DEC_plant *= 3600.0
    
    DEC1 += offset_dec
    #DEC_plant += offset_dec
    
    tree1 = KDTree( numpy.asarray( [RA1, DEC1] ).T )
    tree2 = KDTree( numpy.asarray( [RA2, DEC2] ).T )    
    #tree3 = KDTree( numpy.asarray( [RA_plant, DEC_plant] ).T )    
    
    match_pairs = tree1.query_ball_tree( tree2, r=math.ceil( dt_ratio*0.04 ) )
    #plant_pairs = tree1.query_ball_tree( tree3, r=20.0*0.04 )

    #print match_pairs

    FILE1 = open('a.reg', 'a')
    FILE2 = open('b.reg', 'a')


    X_all, Y_all = [],[]
    X_all2, Y_all2 = [],[]
    MAG_ALL1, MAG_ALL2 = [],[]
    SN_ALL1, SN_ALL2 = [],[]
    IS_PLANT = []
    OBJ_info = []
    for i in range(0, len(tree1.data)):
        
        for j in match_pairs[i]:

                if  abs(MAG1[i] - MAG2[j]) <= 1.0:
                        #print X1[i], Y1[i], X2[j], Y2[j]
                        FILE1.write('circle(%s,%s,5) # text = {%s}\n'%( X1[i], Y1[i], count ) )
                        FILE1.write('circle(%s,%s,5) # text = {%s}\n'%( X2[j], Y2[j], count ) )
                        FILE1.write('line %s %s %s %s\n'%( X1[i], Y1[i], X2[j], Y2[j] ) )
                        FILE2.write('circle(%s,%s,20) # text = {%s}\n'%( X2[j], Y2[j], count ) )
                        #X_all.append( X1[i] )
                        #Y_all.append( Y1[i] )
                        #X_all2.append( X2[i] )
                        #Y_all2.append( Y2[i] )
                        #MAG_ALL1.append( MAG1[i] )
                        #MAG_ALL2.append( MAG2[j] )
                        #SN_ALL1.append( SN1[i] )
                        #SN_ALL2.append( SN2[j] )


                        OBJ_info.append( [ X1[i], Y1[i],
                                           cRA10[i], cDEC10[i],
                                           MAG1[i],
                                           X2[j], Y2[j],
                                           cRA20[j], cDEC20[j],
                                           MAG2[j], SN1[i], SN2[j] ] )
    
                        
                        #if len(plant_pairs[i]) != 0:
                        #        print 'PLANT:',X1[i],Y1[i], '%.3f'%(tree3.query([ RA1[i],DEC1[i] ])[0] / 0.04)
                        #        print MAG1[i], MAG_plant[  tree3.query([ RA1[i],DEC1[i] ])[1] ] - MAG1[i], MAG_plant[  tree3.query([ RA1[i],DEC1[i] ])[1] ] - MAG2[i]
                        #        IS_PLANT.append( 1 )
                        #else:
                        #        print 'NOT PLANT:',X1[i],Y1[i], '%.3f'%(tree3.query([ RA1[i],DEC1[i] ])[0] / 0.04)
                        #        IS_PLANT.append( 0 )
                        
                        #retval = topng(imname1,  X1[i], Y1[i], imname2,  X2[j], Y2[j], 'junk1.fits', 'junk2.fits',ax1, ax2)
                        #if retval:
                        #     pylab.savefig('%s_%s_%s_good.png'%(str(count).rjust(3,'0'), int(round(X1[i])), int(round(Y1[i]))), bbox_inches='tight', pad_inches=0, dpi=120)
                        #else:
                        #    pylab.savefig('%s_%s_%s_bad.png'%(str(count).rjust(3,'0'), int(round(X1[i])), int(round(Y1[i]))), bbox_inches='tight', pad_inches=0, dpi=120)
                        #if IS_PLANT[-1]:
                        #    ax2.annotate( xy=(10,10), s=MAG_plant[  tree3.query([ RA1[i],DEC1[i] ])[1] ], color='w' ) 
                        #
                        #    pylab.savefig('%s_%s_%sP.png'%(str(count).rjust(3,'0'), int(round(X1[i])), int(round(Y1[i]))), bbox_inches='tight', pad_inches=0, dpi=120)
                        #else:
                        #    pylab.savefig('%s_%s_%sN.png'%(str(count).rjust(3,'0'), int(round(X1[i])), int(round(Y1[i]))), bbox_inches='tight', pad_inches=0, dpi=120)
                                                        
    FILE1.close()
    FILE2.close()

    #MAG_ALL1, SN_ALL1, MAG_ALL2, SN_ALL2 = numpy.asarray( MAG_ALL1 ), numpy.asarray( SN_ALL1 ), numpy.asarray( MAG_ALL2 ), numpy.asarray( SN_ALL2 )
    #IS_PLANT = numpy.asarray( IS_PLANT )
    #ok = numpy.where( numpy.asarray( IS_PLANT ) )

    return OBJ_info #X_all, Y_all, X_all2, Y_all2, SN_ALL1, SN_ALL2

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
            str('%d' % degR).rjust(2, '0'), str('%d' % mntR).rjust(2, '0'), str('%.4f' % (secR)).rjust(7, '0') )
    else:
        return '%s%s:%s:%s' % (
            Dsign, str('%d' % degR).rjust(2, '0'), str('%d' % mntR).rjust(2, '0'), str('%.3f' % (secR)).rjust(6, '0') )



        
def write2obj(OBJ_info, imnum, chip, count, NAME):
    #    1 19100001 1549.3  217.2 18:45:12.9675 -20:51:25.431  29.902 1553.9  206.5 18:45:12.3378 -20:51:25.886  29.035

    
    RA1_s  = decdeg2dms( float(OBJ_info[2]), hrs=True )
    DEC1_s = decdeg2dms( float(OBJ_info[3]), hrs=False )
    RA2_s  = decdeg2dms( float(OBJ_info[7]), hrs=True )
    DEC2_s = decdeg2dms( float(OBJ_info[8]), hrs=False )

    print OBJ_info[2], OBJ_info[3], OBJ_info[7], OBJ_info[8]
        
    FILE = open('icii%s_%s_AP2.obj'%(imnum, chip), 'a')
    FILE.write( '%s %s %s %s %s %s %s %s %s %s %s %s\n'%(
        str(count).rjust(5, ' '),
        NAME,
        str('%.1f'%(OBJ_info[0])).rjust(6, ' '),
        str('%.1f'%(OBJ_info[1])).rjust(6, ' '),
        RA1_s, DEC1_s,
        str('%.3f'%(OBJ_info[4])).rjust(7, ' '),
        str('%.1f'%(OBJ_info[5])).rjust(6, ' '),
        str('%.1f'%(OBJ_info[6])).rjust(6, ' '),
        RA2_s, DEC2_s,
        str('%.3f'%(OBJ_info[9])).rjust(7, ' ') ) )

    FILE.close()

def d2a( d ):
    
    s = [0,1,2,3,4,5,6,7,8,9,
         'A','B','C','D','E','F',
         'G','H','I','J','K','L',
         'M','N','O','P','Q','R',
         'S','T','U','V','W','X',
         'Y','Z']
    st = []
    for s2 in s:
        for s3 in s:
           st.append( '%s%s'%( s2, s3 ) ) 
    
    return st[d]
    

def get_fields(chip):
    a_fields = numpy.arange(1,40,2)
    #b_fields = a_fields + 1

    for i in a_fields:
        dir_a = 'icii%s'%(str(i).rjust(2, '0'))
        dir_b = 'icii%s'%(str(i+1).rjust(2, '0'))

        ims1 = commands.getoutput('ls %s/%s_%s_???.fits'%(dir_a, dir_a, chip)).split('\n')
        ims2 = commands.getoutput('ls %s/%s_%s_???.fits'%(dir_b, dir_b, chip)).split('\n')
    
if __name__ == "__main__":


    #a_fields = numpy.arange(1,40,2)
    #b_fields = a_fields + 1
    
    os.system('rm a.reg b.reg')
    
    field1, field2 = str(sys.argv[1]).rjust(2, '0'),  str(sys.argv[2]).rjust(2, '0')
    os.system('rm icii%s_%s_AP2.obj'%( field1, sys.argv[3] ) )


    chip = int( sys.argv[3] )
    #ims1 = commands.getoutput('ls icii%s_%s_???.fits | grep -v \'cel\''%(field1, chip)).split('\n') #[0:5]
    #ims2 = commands.getoutput('ls icii%s_%s_???.fits | grep -v \'cel\''%(field2, chip)).split('\n') #[0:5] 

    ims1 = commands.getoutput('ls icii%s_???.shifts.fits | grep -v \'cel\''%(field1)).split('\n') #[0:5]
    ims2 = commands.getoutput('ls icii%s_???.shifts.fits | grep -v \'cel\''%(field2)).split('\n') #[0:5] 

    
    print ' '.join(ims1)
    print
    print ' '.join(ims2)
    print
    
    cat1 = get_data( ims1 )
    cat2 = get_data( ims2 )

    print cat1[ims1[0]]

    FILE = open('icii%s_icii%s_all.shear'%( field1, field2), 'r')
    dat  = FILE.readlines()
    FILE.close()

    dras, ddecs = [],[]
    for k in dat:
        k = k.strip().split()
        dras.append( float(k[3]) )
        ddecs.append( float(k[4]))

    max_delta = 0.0
    for i in range(0, len(dras)):
        min_delta = 1E9
        for j in range(0, len(dras)):
            if j == i:
                continue
            delta_ij = ( (dras[i] - dras[j])**2 + ( ddecs[i] - ddecs[j])**2 )**0.5
            if delta_ij < min_delta:
                min_delta = delta_ij
        if min_delta > max_delta:
            max_delta = min_delta

    print 'RADIUS OF SEARCH REGION:', 0.5*max_delta, 0.5 * max_delta / 0.04
                
    pylab.ion()
    X_all1, Y_all1 = [],[]
    X_all2, Y_all2 = [],[]
    flux1, flux2 = [],[]
    fig = pylab.figure(figsize=(10,10),frameon=False, facecolor='k', edgecolor='k')
    
    ax1 = fig.add_axes([0,0.5,0.5,0.5],axis_bgcolor='k', frameon=True)
    ax2 = fig.add_axes([0.5,0.5,0.5,0.5],axis_bgcolor='k', frameon=True)
    ax3 = fig.add_axes([0,0,0.5,0.5],axis_bgcolor='k', frameon=True)
    ax4 = fig.add_axes([0.5,0,0.5,0.5],axis_bgcolor='k', frameon=True)


    obj_all = []
    
    count = 0

    #delta = 8.0
    #if len(ims1) > 30:
    #    delta = 4.0

    delta = max_delta / 0.04
        
    for i in range(0, len(ims1)):
        L = dat[i].strip().split()
        OBJ_info = match_radec( cat1[ims1[i]], cat2[ims2[i]], float(L[3]), float(L[4]), delta, i, ims1[i], ims2[i], ax1, ax2)
        
        count2 = 0
        #OBJ_new = []
        for L2 in OBJ_info:
            NAME = '%s%s%sAP%s%s'%( field1, chip, str(i).rjust(3, '0'), str(i).rjust(2, '0'), d2a(count2))
            
        #    write2obj(L2, field1, chip, count, NAME)
            retval = topng(ims1[i], L2[0], L2[1], ims2[i],   L2[5], L2[6], 'junk1.fits', 'junk2.fits', ax1, ax2, ax3, ax4)
            pylab.savefig('%s.png'%(NAME), bbox_inches='tight', pad_inches=0, dpi=72)
            
            count += 1
            count2 += 1
            L3 = L2[:]
            L3.append( NAME )
            obj_all.append( L3 )
            print L3

        #obj_all = numpy.append( obj_all, OBJ_new )

    print obj_all
    
    keep = filter_obj_info( numpy.asarray(obj_all) )
    os.system('mkdir detections_%s'%(chip))

    count = 0
    for k in keep:
        
        write2obj( numpy.asarray(k[:-1], dtype='float'), field1, chip, count, k[-1])
        #retval = topng(ims1[i], L2[0], L2[1], ims2[i],   L2[5], L2[6], 'junk1.fits', 'junk2.fits', ax1, ax2, ax3, ax4)
        #pylab.savefig('%s.png'%(NAME), bbox_inches='tight', pad_inches=0, dpi=72)
        os.system('cp %s.png detections_%s/'%( k[-1], chip))
        count += 1
    
    os.system('rm %s%s*AP*.png'%(field1, chip))
    
