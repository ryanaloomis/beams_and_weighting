tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r2_c0.02_pcwdF',
  specmode = 'cube',
  interactive = True,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 2.,
  perchanweightdensity=False)

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r1_c0.02_pcwdF',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 1,
  perchanweightdensity=False,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r0.5_c0.02_pcwdF',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 0.5,
  perchanweightdensity=False,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r0_c0.02_pcwdF',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 0.,
  perchanweightdensity=False,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r-1_c0.02_pcwdF',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = -1.,
  perchanweightdensity=False,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r-2_c0.02_pcwdF',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = -2.,
  perchanweightdensity=False,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')




tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r2_c0.02_pcwdT',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 2.,
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r1_c0.02_pcwdT',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 1,
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdT.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r0.5_c0.02_pcwdT',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 0.5,
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r0_c0.02_pcwdT',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = 0.,
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r-1_c0.02_pcwdT',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = -1.,
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')

tclean(vis = ['J1610_spw21.ms'],
  imagename = 'r-2_c0.02_pcwdT',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'briggs',
  robust = -2.,
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')




tclean(vis = ['test_r0_c0.02_mod.ms'],
  imagename = 'r0_c0.02_pcwdMOD',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'natural',
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')



tclean(vis = ['test_r-2_c0.02_mod.ms'],
  imagename = 'r-2_c0.02_pcwdMOD',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'natural',
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')


tclean(vis = ['test_r0.5_c0.02_mod.ms'],
  imagename = 'r0.5_c0.02_pcwdMOD',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'natural',
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')


tclean(vis = ['test_r1_c0.02_mod.ms'],
  imagename = 'r1_c0.02_pcwdMOD',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'natural',
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')


tclean(vis = ['test_r2_c0.02_mod.ms'],
  imagename = 'r2_c0.02_pcwdMOD',
  specmode = 'cube',
  interactive = False,
  niter=20000,
  imsize = [1024, 1024],
  cell = '0.02arcsec',
  weighting = 'natural',
  perchanweightdensity=True,
  mask='r2_c0.02_pcwdF.mask',
  threshold='0.0003Jy')
