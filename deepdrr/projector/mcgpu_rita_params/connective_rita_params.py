import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database)]
#[SAMPLING DATA FROM COMMON/CGRA/: X, P, A, B, ITL, ITU] <- ITL is lower bound for "hinted" binary search, ITU is upper bound for "hinted" binary search

connective_Woodard_RITA_PARAMS = np.array([
	[  0.0000000000E+00,  0.0000000000E+00, -2.6048067604E-02, -1.7758144810E-04,   1,   2],
	[  2.3044256386E-03,  1.3716939803E-02, -2.5836200792E-02, -5.8468072867E-05,   1,   3],
	[  4.6088512771E-03,  2.6725953419E-02, -4.9447264774E-02, -1.0344642550E-04,   2,   3],
	[  9.2177025542E-03,  5.0813016085E-02, -9.0501535501E-02, -9.7764260783E-05,   2,   4],
	[  1.8435405108E-02,  9.2447496446E-02, -8.3217200307E-02,  5.6147631027E-05,   3,   4],
	[  2.7653107663E-02,  1.2715784944E-01, -7.6699240145E-02,  1.3932547144E-04,   3,   4],
	[  3.6870810217E-02,  1.5654687538E-01, -7.0908421433E-02,  1.9908602894E-04,   3,   5],
	[  4.6088512771E-02,  1.8177041331E-01, -6.5733650556E-02,  2.4551477098E-04,   4,   5],
	[  5.5306215325E-02,  2.0367973168E-01, -6.1075980614E-02,  2.8168107401E-04,   4,   5],
	[  6.4523917880E-02,  2.2291423801E-01, -5.6856926668E-02,  3.0907303647E-04,   4,   5],
	[  7.3741620434E-02,  2.3996265230E-01, -5.3015296760E-02,  3.2881382922E-04,   4,   5],
	[  8.2959322988E-02,  2.5520429763E-01, -4.9502954225E-02,  3.4193110443E-04,   4,   6],
	[  9.2177025542E-02,  2.6893761280E-01, -8.8649374202E-02,  1.4033015600E-03,   5,   6],
	[  1.1061243065E-01,  2.9278356084E-01, -7.8163021245E-02,  1.3949188259E-03,   5,   6],
	[  1.2904783576E-01,  3.1290638715E-01, -6.9142647147E-02,  1.3406450674E-03,   5,   6],
	[  1.4748324087E-01,  3.3024898581E-01, -6.1360577218E-02,  1.2585610799E-03,   5,   6],
	[  1.6591864598E-01,  3.4546446670E-01, -5.4634521790E-02,  1.1617944981E-03,   5,   7],
	[  1.8435405108E-01,  3.5901781077E-01, -9.3469022504E-02,  4.0286224651E-03,   6,   7],
	[  2.2122486130E-01,  3.8240741228E-01, -7.6019241795E-02,  3.2525216693E-03,   6,   7],
	[  2.5809567152E-01,  4.0223515858E-01, -6.2662619853E-02,  2.5840182920E-03,   6,   8],
	[  2.9496648174E-01,  4.1957195125E-01, -5.2405720875E-02,  2.0382717850E-03,   7,   8],
	[  3.3183729195E-01,  4.3508651479E-01, -4.4485264255E-02,  1.6048535643E-03,   7,   8],
	[  3.6870810217E-01,  4.4921668761E-01, -3.8325127806E-02,  1.2652994661E-03,   7,   8],
	[  4.0557891239E-01,  4.6226067957E-01, -3.3494958107E-02,  1.0007602674E-03,   7,   9],
	[  4.4244972260E-01,  4.7442851328E-01, -5.7641277189E-02,  2.8475936900E-03,   8,   9],
	[  5.1619134304E-01,  4.9670559496E-01, -4.7183681709E-02,  1.8293949938E-03,   8,  10],
	[  5.8993296347E-01,  5.1686443774E-01, -4.0244137131E-02,  1.1952354841E-03,   9,  10],
	[  6.6367458391E-01,  5.3539140507E-01, -3.5490559274E-02,  7.9124270118E-04,   9,  10],
	[  7.3741620434E-01,  5.5259771064E-01, -3.2133506143E-02,  5.2790929387E-04,   9,  11],
	[  8.1115782477E-01,  5.6869476031E-01, -2.9692919914E-02,  3.5250401333E-04,  10,  11],
	[  8.8489944521E-01,  5.8383367811E-01, -5.4276253124E-02,  8.7512948135E-04,  10,  12],
	[  1.0323826861E+00,  6.1166352875E-01, -4.8862126683E-02,  2.6997747238E-04,  11,  12],
	[  1.1798659269E+00,  6.3674908995E-01, -4.5891938505E-02,  3.3672664368E-05,  11,  13],
	[  1.3273491678E+00,  6.5952735855E-01, -4.3793751891E-02, -9.0597456679E-05,  12,  13],
	[  1.4748324087E+00,  6.8030798586E-01, -4.2196590543E-02, -1.5714404554E-04,  12,  14],
	[  1.6223156495E+00,  6.9933343227E-01, -4.0904488662E-02, -1.9255801669E-04,  13,  14],
	[  1.7697988904E+00,  7.1680156126E-01, -7.6460882658E-02, -8.5990878839E-04,  13,  14],
	[  2.0647653722E+00,  7.4770580764E-01, -3.7976036484E-02, -2.2063336886E-04,  13,  15],
	[  2.2122486130E+00,  7.6140679544E-01, -3.7395078099E-02,  7.0290637029E-05,  14,  15],
	[  2.3597318539E+00,  7.7408979964E-01, -6.9149843207E-02, -8.3703495228E-04,  14,  16],
	[  2.6546983356E+00,  7.9680159809E-01, -3.4545716555E-02, -2.0028032778E-04,  15,  16],
	[  2.8021815765E+00,  8.0698725045E-01, -3.3968376062E-02, -5.6501512877E-05,  15,  17],
	[  2.9496648174E+00,  8.1648256664E-01, -6.3988754878E-02, -7.2361175640E-04,  16,  17],
	[  3.2446312991E+00,  8.3363896111E-01, -6.1930024389E-02, -6.7402081486E-04,  16,  18],
	[  3.5395977808E+00,  8.4866969351E-01, -5.9995767291E-02, -6.2801171642E-04,  17,  18],
	[  3.8345642626E+00,  8.6189635247E-01, -5.8172139918E-02, -5.8581506905E-04,  17,  19],
	[  4.1295307443E+00,  8.7358376519E-01, -1.0658127418E-01, -2.1195167938E-03,  18,  19],
	[  4.7194637078E+00,  8.9318208653E-01, -1.0091763338E-01, -1.8619657653E-03,  18,  19],
	[  5.3093966712E+00,  9.0882155594E-01, -9.5781136797E-02, -1.6463573000E-03,  18,  20],
	[  5.8993296347E+00,  9.2146296152E-01, -9.1105126231E-02, -1.4645069040E-03,  19,  20],
	[  6.4892625982E+00,  9.3179952933E-01, -8.6833622771E-02, -1.3099550184E-03,  19,  20],
	[  7.0791955617E+00,  9.4033997562E-01, -8.2918981169E-02, -1.1776621570E-03,  19,  21],
	[  7.6691285251E+00,  9.4746338894E-01, -7.9320419589E-02, -1.0636791402E-03,  20,  21],
	[  8.2590614886E+00,  9.5345623490E-01, -1.4062564628E-01, -3.6890814176E-03,  20,  22],
	[  9.4389274155E+00,  9.6287777576E-01, -1.3050625769E-01, -3.0818992162E-03,  21,  22],
	[  1.0618793342E+01,  9.6983700549E-01, -1.2166380651E-01, -2.6100681067E-03,  21,  23],
	[  1.1798659269E+01,  9.7510102471E-01, -1.1388402445E-01, -2.2373444071E-03,  22,  23],
	[  1.2978525196E+01,  9.7916495589E-01, -1.0699628099E-01, -1.9386087285E-03,  22,  24],
	[  1.4158391123E+01,  9.8235853617E-01, -1.0086336499E-01, -1.6960425742E-03,  23,  25],
	[  1.5338257050E+01,  9.8490743075E-01, -9.5373874549E-02, -1.4967538392E-03,  24,  25],
	[  1.6518122977E+01,  9.8696982454E-01, -1.6484326587E-01, -5.0502887214E-03,  24,  26],
	[  1.8877854831E+01,  9.9005752950E-01, -1.5068283732E-01, -4.1013271132E-03,  25,  26],
	[  2.1237586685E+01,  9.9221303637E-01, -1.3873019897E-01, -3.4058480150E-03,  25,  26],
	[  2.3597318539E+01,  9.9376956788E-01, -2.2481328708E-01, -1.0725812623E-02,  25,  27],
	[  2.8316782247E+01,  9.9580419572E-01, -1.9959151979E-01, -8.1039097142E-03,  26,  27],
	[  3.3036245954E+01,  9.9702346340E-01, -1.7952247961E-01, -6.3800335741E-03,  26,  28],
	[  3.7755709662E+01,  9.9780358506E-01, -1.6322254085E-01, -5.1774588928E-03,  27,  28],
	[  4.2475173370E+01,  9.9832838192E-01, -1.4973538581E-01, -4.2997664711E-03,  27,  29],
	[  4.7194637078E+01,  9.9869567714E-01, -2.3909327275E-01, -1.3543600312E-02,  28,  29],
	[  5.6633564493E+01,  9.9915812833E-01, -2.1214015339E-01, -1.0195629479E-02,  28,  29],
	[  6.6072491909E+01,  9.9942331228E-01, -1.9080013814E-01, -7.9766975058E-03,  28,  30],
	[  7.5511419324E+01,  9.9958687615E-01, -2.8721681800E-01, -2.3515688498E-02,  29,  30],
	[  9.4389274155E+01,  9.9976610510E-01, -2.5111283882E-01, -1.6452257964E-02,  29,  31],
	[  1.1326712899E+02,  9.9985450881E-01, -3.4576848306E-01, -4.3442915116E-02,  30,  31],
	[  1.5102283865E+02,  9.9993232563E-01, -4.0959451176E-01, -9.3040322806E-02,  30,  32],
	[  2.2653425797E+02,  9.9997763636E-01, -3.5378993340E-01, -4.8132328435E-02,  31,  32],
	[  3.0204567730E+02,  9.9998996735E-01, -3.7583974670E-01, -3.1751810758E-01,  31,  32],
	[  6.0409135459E+02,  9.9999859775E-01, -3.7092727970E-01, -3.2748529768E-01,  31,  33],
	[  1.2081827092E+03,  9.9999980933E-01, -3.7112273636E-01, -3.2932624691E-01,  32,  33],
	[  2.4163654184E+03,  9.9999997423E-01, -3.7516194292E-01, -3.2539443082E-01,  32,  33],
	[  4.8327308368E+03,  9.9999999649E-01, -3.8245636056E-01, -3.1682723885E-01,  32,  34],
	[  9.6654616735E+03,  9.9999999951E-01, -3.9278586995E-01, -3.0403780531E-01,  33,  34],
	[  1.9330923347E+04,  9.9999999993E-01, -4.0593230028E-01, -2.8724241565E-01,  33,  34],
	[  3.8661846694E+04,  9.9999999999E-01, -4.2133040634E-01, -2.6694087912E-01,  33,  35],
	[  7.7323693388E+04,  1.0000000000E+00, -4.3785148241E-01, -2.4425863985E-01,  34,  35],
	[  1.5464738678E+05,  1.0000000000E+00, -4.5390472158E-01, -2.2096187235E-01,  34,  35],
	[  3.0929477355E+05,  1.0000000000E+00, -4.6791807739E-01, -1.9903494885E-01,  34,  36],
	[  6.1858954711E+05,  1.0000000000E+00, -4.7892052513E-01, -1.8003908918E-01,  35,  36],
	[  1.2371790942E+06,  1.0000000000E+00, -4.8678678112E-01, -1.6470023499E-01,  35,  37],
	[  2.4743581884E+06,  1.0000000000E+00, -4.9201885259E-01, -1.5295352070E-01,  36,  37],
	[  4.9487163768E+06,  1.0000000000E+00, -4.9534023217E-01, -1.4427184332E-01,  36,  37],
	[  9.8974327537E+06,  1.0000000000E+00, -4.9740238763E-01, -1.3799169672E-01,  36,  38],
	[  1.9794865507E+07,  1.0000000000E+00, -4.9867925962E-01, -1.3350225121E-01,  37,  38],
	[  3.9589731015E+07,  1.0000000000E+00, -4.9947804455E-01, -1.3031230490E-01,  37,  38],
	[  7.9179462029E+07,  1.0000000000E+00, -4.9998635729E-01, -1.2805228428E-01,  37,  39],
	[  1.5835892406E+08,  1.0000000000E+00, -5.0031609751E-01, -1.2645318200E-01,  38,  39],
	[  3.1671784812E+08,  1.0000000000E+00, -5.0053395800E-01, -1.2532231949E-01,  38,  40],
	[  6.3343569624E+08,  1.0000000000E+00, -5.0068021537E-01, -1.2452273041E-01,  39,  40],
	[  1.2668713925E+09,  1.0000000000E+00, -4.0286980843E-01, -4.1361124799E-02,  39,  41],
	[  1.9003070887E+09,  1.0000000000E+00, -3.2275062900E-01, -2.0644282192E-02,  40,  41],
	[  2.5337427849E+09,  1.0000000000E+00, -2.6722777504E-01, -1.2372754146E-02,  40,  41],
	[  3.1671784812E+09,  1.0000000000E+00, -2.2746254972E-01, -8.2419661805E-03,  40,  42],
	[  3.8006141774E+09,  1.0000000000E+00, -1.9779949164E-01, -5.8835788806E-03,  41,  43],
	[  4.4340498736E+09,  1.0000000000E+00, -1.7489251666E-01, -4.4105805200E-03,  42,  44],
	[  5.0674855699E+09,  1.0000000000E+00, -1.5669663242E-01, -3.4291132135E-03,  43,  44],
	[  5.7009212661E+09,  1.0000000000E+00, -1.4190619753E-01, -2.7423935405E-03,  43,  45],
	[  6.3343569624E+09,  1.0000000000E+00, -1.2965303902E-01, -2.2431499163E-03,  44,  45],
	[  6.9677926586E+09,  1.0000000000E+00, -1.1933907226E-01, -1.8688389501E-03,  44,  46],
	[  7.6012283548E+09,  1.0000000000E+00, -1.1053955151E-01, -1.5809890811E-03,  45,  46],
	[  8.2346640511E+09,  1.0000000000E+00, -1.0294483693E-01, -1.3548779041E-03,  45,  47],
	[  8.8680997473E+09,  1.0000000000E+00, -9.6324035345E-02, -1.1740292758E-03,  46,  48],
	[  9.5015354435E+09,  1.0000000000E+00, -9.0501546622E-02, -1.0271191842E-03,  47,  48],
	[  1.0134971140E+10,  1.0000000000E+00, -8.5341492916E-02, -9.0615632103E-04,  47,  48],
	[  1.0768406836E+10,  1.0000000000E+00, -8.0737114386E-02, -8.0537056390E-04,  47,  49],
	[  1.1401842532E+10,  1.0000000000E+00, -7.6603381788E-02, -7.2051116721E-04,  48,  49],
	[  1.2035278228E+10,  1.0000000000E+00, -7.2871744968E-02, -6.4839070535E-04,  48,  50],
	[  1.2668713925E+10,  1.0000000000E+00, -6.9486331377E-02, -5.8658110748E-04,  49,  50],
	[  1.3302149621E+10,  1.0000000000E+00, -6.6401148888E-02, -5.3320645368E-04,  49,  51],
	[  1.3935585317E+10,  1.0000000000E+00, -6.3577996899E-02, -4.8679886258E-04,  50,  52],
	[  1.4569021013E+10,  1.0000000000E+00, -6.0984885246E-02, -4.4619643090E-04,  51,  53],
	[  1.5202456710E+10,  1.0000000000E+00, -5.8594822687E-02, -4.1046976254E-04,  52,  54],
	[  1.5835892406E+10,  1.0000000000E+00, -5.6384878061E-02, -3.7886828708E-04,  53,  55],
	[  1.6469328102E+10,  1.0000000000E+00, -5.4335445184E-02, -3.5078050189E-04,  54,  56],
	[  1.7102763798E+10,  1.0000000000E+00, -5.2429661750E-02, -3.2570415788E-04,  55,  58],
	[  1.7736199495E+10,  1.0000000000E+00, -5.0652945884E-02, -3.0322364496E-04,  57,  60],
	[  1.8369635191E+10,  1.0000000000E+00, -4.8992623503E-02, -2.8299265629E-04,  59,  63],
	[  1.9003070887E+10,  1.0000000000E+00, -4.7437626382E-02, -2.6472076861E-04,  62, 128],
	[  1.9636506583E+10,  1.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00, 127, 128],
])