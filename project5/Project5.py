import pandas as pd
from pandas import DataFrame
from scipy.stats import ttest_ind

train_data_path = "训练集.xlsx"
train_label_path = "训练集类别.csv"


def loadData():
    df = pd.read_excel(train_data_path, header=None, index_col=None)
    df = pd.DataFrame(df.values[:, 1:].T, columns=df[0])
    df = df.drop("Gene_ID", axis=1)

    labels = pd.read_csv(train_label_path)
    labels = labels.drop('Sample', axis=1)
    return df.join(labels)


def tTestSelection(df: DataFrame, n=150):
    g = df.groupby("Class")
    columns = list(df.columns)
    columns.remove("Class")
    t_values = []
    for name in columns:
        l = g[name].apply(lambda x: list(x))
        (t, p) = ttest_ind(*l)
        t_values.append((name, t))
    t_values.sort(key=lambda x: x[1], reverse=True)
    return t_values[:n]


def feature():
    df = loadData()
    print("Loaded.")
    t_values = tTestSelection(df, 150)
    print(t_values)


features = [('A_24_P500584_SpotID_3653799', 35.466417065753625), ('Hs456200.1_SpotID_3659465', 13.795743044882538),
            ('A_23_P125519_SpotID_3658212', 6.403747877977484), ('A_23_P315345_SpotID_3653219', 6.090437923117008),
            ('A_24_P237389_SpotID_3662821', 5.611977107813027), ('A_23_P217409_SpotID_3656884', 5.366236451118951),
            ('A_24_P134653_SpotID_3660765', 4.492581505027911), ('A_23_P148255_SpotID_3653391', 4.23018824244875),
            ('Hs290759.1_SpotID_3660024', 3.9766119704443668), ('A_24_P79529_SpotID_3661402', 3.9452151856314153),
            ('A_32_P183001_SpotID_3662637', 3.816041570720035), ('A_24_P942604_SpotID_3656547', 3.517313142329783),
            ('A_23_P96291_SpotID_3662423', 3.3231923875147302), ('Hs433518.5_SpotID_3660741', 3.1766216933670948),
            ('A_23_P256021_SpotID_3653610', 3.1632331575230506), ('A_23_P136870_SpotID_3653978', 3.1526386808778004),
            ('A_23_P127557_SpotID_3661308', 3.084382928191403), ('A_24_P119609_SpotID_3657955', 2.997583529801981),
            ('A_32_P166921_SpotID_3655984', 2.917487198350365), ('A_23_P85053_SpotID_3657361', 2.8944652043525427),
            ('A_23_P64611_SpotID_3660446', 2.8633599347949836), ('A_32_P384716_SpotID_3660029', 2.851850183996635),
            ('A_23_P103775_SpotID_3662774', 2.845362052412128), ('A_23_P167212_SpotID_3653579', 2.8368554938145514),
            ('A_24_P156576_SpotID_3658718', 2.824053251131163), ('A_23_P45396_SpotID_3658883', 2.8068541124021493),
            ('A_23_P9836_SpotID_3654953', 2.7928275082760448), ('A_23_P148541_SpotID_3658039', 2.789546202699387),
            ('A_32_P42964_SpotID_3659733', 2.75596090747537), ('A_23_P115064_SpotID_3661601', 2.7463114761617478),
            ('A_23_P161064_SpotID_3658962', 2.7373598248940456), ('A_23_P121499_SpotID_3661588', 2.714290215420465),
            ('Hs279813.17_SpotID_3657070', 2.664503641036226), ('A_23_P146997_SpotID_3657483', 2.6603328134899207),
            ('Hs406565.493_SpotID_3659574', 2.629204740579596), ('A_24_P937193_SpotID_3658742', 2.628690893387535),
            ('Hs282793.1_SpotID_3655969', 2.614536038531086), ('A_23_P133279_SpotID_3658563', 2.5949343383874828),
            ('A_24_P188231_SpotID_3657497', 2.592056045050497), ('A_23_P409951_SpotID_3662036', 2.5902618888326354),
            ('r60_a135_SpotID_3662791', 2.5859120515289087), ('Hs75438.5_SpotID_3656782', 2.571985934304141),
            ('A_23_P130418_SpotID_3659241', 2.5495215937443825), ('A_23_P108200_SpotID_3654062', 2.5489690698806866),
            ('A_32_P4700_SpotID_3655174', 2.5306318532088583), ('A_32_P174214_SpotID_3655007', 2.530331918155426),
            ('A_24_P398572_SpotID_3655429', 2.522687160273054), ('A_23_P161596_SpotID_3654484', 2.5185090505195005),
            ('A_23_P103256_SpotID_3652452', 2.514554752622382), ('Hs49573.3_SpotID_3655020', 2.509082326623579),
            ('A_23_P93009_SpotID_3661794', 2.5036059060928246), ('A_23_P202905_SpotID_3659657', 2.496596741075647),
            ('A_23_P44794_SpotID_3652944', 2.481047107557529), ('A_23_P41645_SpotID_3662988', 2.46858640852406),
            ('A_23_P46903_SpotID_3653762', 2.464964521493492), ('Hs148844.1_SpotID_3654647', 2.4586573585206346),
            ('A_23_P151895_SpotID_3657006', 2.435759481614264), ('A_23_P104804_SpotID_3661966', 2.4346508955073567),
            ('A_24_P311917_SpotID_3653339', 2.426610802060747), ('A_23_P127117_SpotID_3655527', 2.419818138028449),
            ('A_23_P133585_SpotID_3661790', 2.418269315067506), ('A_23_P159053_SpotID_3656790', 2.4059246354020307),
            ('A_32_P55979_SpotID_3652739', 2.3883912735175503), ('Hs7181.2_SpotID_3661992', 2.38624064959637),
            ('A_23_P59807_SpotID_3657392', 2.3784953471517962), ('A_23_P11144_SpotID_3653627', 2.3750577687955974),
            ('A_23_P89283_SpotID_3662128', 2.3606680352019676), ('A_24_P827037_SpotID_3661965', 2.355934557403714),
            ('A_23_P7732_SpotID_3660994', 2.354926911397135), ('A_23_P418451_SpotID_3662481', 2.350962515163163),
            ('A_23_P256470_SpotID_3653922', 2.3475981645618145), ('A_24_P237374_SpotID_3657175', 2.347357147418089),
            ('A_23_P356554_SpotID_3657147', 2.3444496680383327), ('A_32_P108738_SpotID_3654027', 2.3395311638307015),
            ('A_24_P152315_SpotID_3658313', 2.3382631843551214), ('A_23_P337168_SpotID_3654641', 2.3127692642578994),
            ('A_32_P209094_SpotID_3657827', 2.303960945216089), ('A_23_P46378_SpotID_3660790', 2.301449457334705),
            ('A_24_P141332_SpotID_3653719', 2.296251804322326), ('A_23_P20303_SpotID_3662074', 2.295312566766865),
            ('A_24_P124370_SpotID_3658157', 2.2929595998133507), ('A_23_P212436_SpotID_3661640', 2.2926129329821676),
            ('A_24_P566932_SpotID_3662125', 2.2701761134569574), ('A_23_P145_SpotID_3660844', 2.2687934655547513),
            ('A_32_P840235_SpotID_3657121', 2.2635528906182705), ('A_24_P148796_SpotID_3653435', 2.260804580300352),
            ('A_23_P76914_SpotID_3653644', 2.259970892000264), ('A_23_P35467_SpotID_3653833', 2.254323286834222),
            ('A_23_P404606_SpotID_3657965', 2.2517937226418248), ('A_32_P59486_SpotID_3663012', 2.250110607002998),
            ('A_23_P75921_SpotID_3658119', 2.249227147280275), ('A_23_P214066_SpotID_3662236', 2.2467340353284095),
            ('A_23_P41970_SpotID_3656913', 2.2415835592897633), ('A_23_P253958_SpotID_3660247', 2.2318037180167067),
            ('A_23_P37441_SpotID_3654989', 2.230502333112305), ('Hs443722.1_SpotID_3654934', 2.217934632605927),
            ('A_32_P40424_SpotID_3656746', 2.2015050256169753), ('A_23_P433111_SpotID_3655201', 2.2007683677056527),
            ('A_32_P486762_SpotID_3661937', 2.1867656785558407), ('A_24_P154037_SpotID_3662636', 2.1835586444707187),
            ('A_23_P162787_SpotID_3661332', 2.182583805618197), ('A_24_P237265_SpotID_3660329', 2.182559458072413),
            ('A_23_P254688_SpotID_3661103', 2.1810947072906686), ('A_23_P21485_SpotID_3656063', 2.179098185861557),
            ('Hs32362.1_SpotID_3659551', 2.171443597958103), ('A_24_P943613_SpotID_3659396', 2.1680454728807836),
            ('A_23_P83751_SpotID_3654786', 2.1590039426656156), ('A_24_P923684_SpotID_3654508', 2.1582744674059073),
            ('A_23_P29225_SpotID_3660807', 2.1569244890545027), ('A_23_P213908_SpotID_3657690', 2.1492825948736343),
            ('A_32_P60185_SpotID_3662230', 2.133465995282954), ('A_23_P133432_SpotID_3659085', 2.124184370160716),
            ('A_23_P64173_SpotID_3662308', 2.1233867296824425), ('A_23_P81770_SpotID_3657946', 2.1130953273172604),
            ('A_23_P138967_SpotID_3655377', 2.110065697058053), ('A_23_P211762_SpotID_3653902', 2.105163041112629),
            ('A_32_P29806_SpotID_3655019', 2.097166855690107), ('A_23_P2492_SpotID_3653839', 2.0956558253081585),
            ('A_32_P181107_SpotID_3655943', 2.0932601308597123), ('A_23_P12044_SpotID_3660656', 2.088964000331428),
            ('A_24_P226322_SpotID_3661423', 2.0866648437292414), ('A_23_P103371_SpotID_3662864', 2.08490908010051),
            ('A_23_P254654_SpotID_3658621', 2.082697571235344), ('A_24_P264928_SpotID_3654161', 2.0799993554902154),
            ('A_23_P148372_SpotID_3658831', 2.078686617305241), ('A_23_P47377_SpotID_3658333', 2.07440823142741),
            ('A_23_P201030_SpotID_3657176', 2.0725258992395634), ('A_23_P209394_SpotID_3659770', 2.070325330319641),
            ('A_23_P75430_SpotID_3654859', 2.064005309914297), ('A_23_P50919_SpotID_3660376', 2.063476339150866),
            ('A_24_P380679_SpotID_3662221', 2.06204454414641), ('A_24_P787914_SpotID_3658800', 2.053215609280605),
            ('A_24_P183664_SpotID_3655539', 2.0486147299667143), ('Hs152925.1_SpotID_3660170', 2.0450935319725154),
            ('A_23_P97112_SpotID_3661555', 2.044042317199424), ('A_23_P390755_SpotID_3659794', 2.043520270805174),
            ('A_23_P156687_SpotID_3656497', 2.042384278261886), ('A_24_P931944_SpotID_3658895', 2.0418898170761426),
            ('A_24_P186994_SpotID_3652629', 2.0410068837932833), ('A_24_P222655_SpotID_3658430', 2.0335468808713353),
            ('A_23_P122959_SpotID_3662264', 2.0323146323899506), ('Hs421376.1_SpotID_3654478', 2.0316648216887647),
            ('A_32_P233769_SpotID_3659972', 2.0272831476270228), ('A_24_P64167_SpotID_3657854', 2.0257136741296846),
            ('A_32_P78816_SpotID_3655706', 2.024319077171871), ('A_24_P412156_SpotID_3658846', 2.022503711109166),
            ('A_23_P83134_SpotID_3661508', 2.01700989824477), ('A_32_P115258_SpotID_3654445', 2.0164085239243015),
            ('A_23_P201035_SpotID_3662662', 2.014885995431415), ('A_32_P2738_SpotID_3652461', 2.0129802425974197)]
