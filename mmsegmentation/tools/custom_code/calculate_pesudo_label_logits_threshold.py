from tqdm import tqdm
import pickle
import torch
import os
import numpy as np
def calculate_class_top_logits(file_dir, class_index, top_ratio):
    logits_list = []
    for idx in tqdm(range(500)):
        sample_logits = pickle.load(open(os.path.join(file_dir, f"{idx}.pkl"), "rb")).cpu()
        sample_logits = sample_logits.squeeze()
        sample_max_logits, sample_pesudo_labels = sample_logits.max(dim=0)
        unique_labels = torch.unique(sample_pesudo_labels).type(torch.int64).cpu().numpy()
        if class_index not in unique_labels:
            continue
        pesudo_label_index_mask = sample_pesudo_labels == class_index
        non_zero_mask = sample_max_logits != 0
        class_logits = sample_max_logits[pesudo_label_index_mask & non_zero_mask]
        class_logits = class_logits.detach().cpu().numpy().tolist()
        logits_list.extend(class_logits)
    logits_list = np.asarray(logits_list)
    logits_sorted = np.sort(logits_list)[::-1]
    top_index = int(len(logits_sorted) * top_ratio)
    if len(logits_sorted) > 0:
        threshold_logit = logits_sorted[top_index]
    else:
        threshold_logit = 0
    return threshold_logit


if __name__ == '__main__':
    top_ratio = 0.5
    print(top_ratio)
    # num_classes = 19 # cityscapes
    # num_classes = 20 # voc
    num_classes = 65 # mapillary
    threshold_logits = []
    dataset='mapillary'
    for class_idx in range(num_classes):
        # threshold_logit = calculate_class_top_logits("./san_prompt_expanded_logits_cityscapes/", class_idx, top_ratio)
        # threshold_logit = calculate_class_top_logits("./san_logits_voc/", class_idx, top_ratio)
        threshold_logit = calculate_class_top_logits(f"./san_prompt_expanded_logits_{dataset}/", class_idx, top_ratio)
        threshold_logits.append(threshold_logit)
    print(threshold_logits)
    # top 5%
    # san_mapillary_logits_threshold = [0.2648111581802368, 1.5551313161849976, 1.4176703691482544, 2.865515947341919, 2.0327341556549072, 1.0000050067901611, 2.7463865280151367, 0.6283930540084839, 0.9684611558914185, 2.2351572513580322, 2.8698179721832275, 0, 4.1455535888671875, 1.4593349695205688, 0.2526891231536865, 3.5620408058166504, 3.50738787651062, 2.234792947769165, 4.072118282318115, 0.9900062084197998, 1.6966696977615356, 0, 0, 0, 0, 2.619001865386963, 2.189218044281006, 2.2109029293060303, 5.255155086517334, 2.0121240615844727, 1.6151020526885986, 1.3258024454116821, 2.444265604019165, 2.0846590995788574, 1.5633811950683594, 3.0306828022003174, 0.9877110123634338, 0.3444611430168152, 2.0680642127990723, 0, 1.3189196586608887, 1.4172338247299194, 1.2321490049362183, 1.8148711919784546, 1.1706451177597046, 1.052062749862671, 2.040908098220825, 1.4163000583648682, 1.8521161079406738, 0, 1.0851484537124634, 3.728440284729004, 1.4674303531646729, 1.4010400772094727, 1.862186074256897, 1.9149545431137085, 1.009891152381897, 1.0323808193206787, 1.611673355102539, 0.9943180084228516, 1.9410858154296875, 1.9196603298187256, 1.0117534399032593, 0.4611998498439789, 1.0573225021362305]

    # top 10%
    # san_cityscapes_logits_threshold = [1.5474570989608765, 2.565410852432251, 1.858138918876648, 1.713448405265808, 3.265012264251709, 1.5394763946533203,
    #  1.4438365697860718, 2.2762832641601562, 1.517593502998352, 1.4485270977020264, 1.571458339691162,
    #  0.9942319393157959, 1.2791723012924194, 2.1838502883911133, 2.9471640586853027, 2.29648756980896,
    #  1.316765546798706, 1.3644033670425415, 1.894166111946106]
    # san_mapillary_logits_threshold = [0.22436702251434326, 1.4874109029769897, 1.4098931550979614, 2.4623138904571533, 1.8445247411727905, 0.9694060683250427, 2.2029740810394287, 0.6281765103340149, 0.9542619585990906, 2.0854125022888184, 2.1052539348602295, 0, 3.4001922607421875, 1.3194788694381714, 0.2462461292743683, 3.0265018939971924, 3.234869956970215, 1.978623867034912, 3.98140811920166, 0.9739317893981934, 1.6802444458007812, 0, 0, 0, 0, 2.1002376079559326, 1.4715594053268433, 2.052940607070923, 4.595953941345215, 1.814181923866272, 1.3829578161239624, 1.2608461380004883, 2.0254364013671875, 2.0202105045318604, 1.3696801662445068, 2.6793212890625, 0.9875099062919617, 0.2764441668987274, 1.9879523515701294, 0, 1.2061890363693237, 1.4136252403259277, 1.201991081237793, 1.4637775421142578, 0.976740300655365, 1.040185809135437, 1.591469168663025, 1.2722021341323853, 1.5090301036834717, 0, 0.9642882943153381, 2.3493685722351074, 1.240419626235962, 1.3102266788482666, 1.3196542263031006, 1.6826509237289429, 1.0018532276153564, 0.9037175178527832, 1.5709786415100098, 0.9903153777122498, 1.6177325248718262, 1.6345003843307495, 0.6839448809623718, 0.45341065526008606, 1.031076431274414]

    # top 15%
    # san_mapillary_logits_threshold = [0.21874262392520905, 1.4301530122756958, 1.4040743112564087, 2.227327585220337, 1.365898609161377, 0.9477671384811401, 1.970780849456787, 0.6279125809669495, 0.9485656023025513, 1.969452142715454, 1.3687779903411865, 0, 3.158936023712158, 1.255296230316162, 0.23872095346450806, 2.7071964740753174, 2.8918986320495605, 1.8471784591674805, 3.3843624591827393, 0.9611904621124268, 1.6568800210952759, 0, 0, 0, 0, 2.093977928161621, 1.3674383163452148, 1.941281795501709, 4.041079044342041, 1.6153289079666138, 1.2296802997589111, 1.2342429161071777, 1.9451433420181274, 1.6000304222106934, 1.3624610900878906, 2.3424949645996094, 0.9873703718185425, 0.2129143476486206, 1.9879051446914673, 0, 1.1431255340576172, 1.409789800643921, 1.1000791788101196, 1.390337586402893, 0.8299326300621033, 0.9962357878684998, 1.3784586191177368, 0.680397093296051, 1.2107144594192505, 0, 0.8838568925857544, 1.6548199653625488, 1.1274867057800293, 1.2199002504348755, 1.2483986616134644, 1.5100971460342407, 0.9935503602027893, 0.8410444259643555, 1.5425279140472412, 0.9825652837753296, 1.4425585269927979, 1.4633172750473022, 0.6780804991722107, 0.4432718753814697, 0.9796928763389587]


    # top 20%
    # san_cityscapes_logits_threshold = [1.1922125816345215, 2.301851749420166, 1.2906343936920166, 1.168750286102295, 1.996658205986023, 0.7570928335189819, 1.2795923948287964, 1.7305257320404053, 1.0150513648986816, 1.3491066694259644, 1.1607956886291504, 0.9820324778556824, 0.7445511221885681, 2.114917755126953, 2.035916566848755, 1.9401479959487915, 0.9684708714485168, 0.8137786984443665, 1.094667673110962]
    # sam_cityscapes_threshold_logits = [0.447265625, 0.381103515625, 0.400146484375, 0.311279296875, 0.48388671875, 0.36572265625, 0.48291015625, 0.421875, 0.43115234375, 0.423095703125, 0.441650390625, 0.4375, 0.368408203125, 0.75439453125, 0.61474609375, 0.78466796875, 0.391357421875, 0.60107421875, 0.625]
    # san_mapillary_logits_threshold = [0.21360555291175842, 1.3920364379882812, 1.3934916257858276, 2.0186831951141357, 0.8594971895217896, 0.9292413592338562, 1.7292989492416382, 0.627479612827301, 0.9433804750442505, 1.7976701259613037, 1.2396548986434937, 0, 3.014939069747925, 1.2149888277053833, 0.23588331043720245, 2.366422176361084, 2.6854677200317383, 1.727423071861267, 3.2665293216705322, 0.9549123048782349, 1.5133744478225708, 0, 0, 0, 0, 2.0868208408355713, 1.3475717306137085, 1.823897361755371, 3.494551420211792, 1.4849668741226196, 1.162499189376831, 1.217604160308838, 1.7809287309646606, 1.2940361499786377, 1.3556252717971802, 2.1788978576660156, 0.9872310161590576, 0.2029213309288025, 1.9876878261566162, 0, 1.0972522497177124, 1.4060982465744019, 0.9823172688484192, 1.382989764213562, 0.7434610724449158, 0.8756065964698792, 1.332170844078064, 0.6711662411689758, 1.084380865097046, 0, 0.7968075275421143, 1.4287432432174683, 1.016462802886963, 1.0873113870620728, 1.1078144311904907, 1.4263243675231934, 0.9846782088279724, 0.7664245367050171, 1.5121369361877441, 0.9695954322814941, 1.3948204517364502, 1.3288357257843018, 0.6720066070556641, 0.4386339783668518, 0.919492781162262]


    # top 30%
    # san_cityscapes_logits_threshold = [1.3172924518585205, 1.9998561143875122, 1.4444035291671753, 1.3216272592544556, 2.345761775970459,
    #  1.2329155206680298, 1.0668926239013672, 1.6766622066497803, 1.1390260457992554, 1.1509145498275757,
    #  1.3897994756698608, 0.9774896502494812, 0.8637236952781677, 1.8247607946395874, 2.355012893676758,
    #  1.7781503200531006, 1.1183868646621704, 0.9356830716133118, 1.4372142553329468]
    # san_mapillary_logits_threshold = [0.20620214939117432, 1.3206487894058228, 1.2162837982177734, 1.6895965337753296, 0.7969141602516174, 0.9187865853309631, 1.5159776210784912, 0.6259702444076538, 0.9326619505882263, 1.5981661081314087, 1.1259740591049194, 0, 2.883073329925537, 1.146687626838684, 0.2303723394870758, 1.9553850889205933, 2.114281415939331, 1.5281277894973755, 2.7219531536102295, 0.938151478767395, 0.8466380834579468, 0, 0, 0, 0, 1.8873968124389648, 1.323807716369629, 1.6293219327926636, 2.6167173385620117, 1.259863257408142, 1.09075927734375, 1.1921565532684326, 1.4042651653289795, 0.7327224016189575, 1.3325457572937012, 1.919741153717041, 0.9870772957801819, 0.19514262676239014, 1.9754544496536255, 0, 0.9892647862434387, 1.3986068964004517, 0.9417718648910522, 1.359868049621582, 0.5714660882949829, 0.6380545496940613, 1.1066077947616577, 0.6241672039031982, 1.0074856281280518, 0, 0.7092477083206177, 1.1859210729599, 0.9117567539215088, 0.6623800992965698, 0.9913734793663025, 1.2648481130599976, 0.9682055115699768, 0.7179420590400696, 1.4227346181869507, 0.9462730288505554, 1.1865183115005493, 1.1808093786239624, 0.6604098081588745, 0.32694211602211, 0.7927626371383667]

    # top 50%
    # san_mapillary_logits_threshold = [0.19050996005535126, 1.1426186561584473, 0.9260540008544922, 1.2563132047653198, 0.5916278958320618, 0.9118586182594299, 1.238429069519043, 0.5806747674942017, 0.8971202373504639, 1.1430928707122803, 0.6792925000190735, 0, 2.5907227993011475, 1.0505191087722778, 0.21792756021022797, 1.5319817066192627, 1.4052793979644775, 1.2772247791290283, 2.2094035148620605, 0.8848975300788879, 0.6872119307518005, 0, 0, 0, 0, 1.496749997138977, 0.8317766785621643, 1.2934184074401855, 2.3122739791870117, 1.0715463161468506, 1.0112758874893188, 1.1272307634353638, 1.1997417211532593, 0.6111717224121094, 0.7698605060577393, 1.3760405778884888, 0.9864566922187805, 0.18862739205360413, 1.2139678001403809, 0, 0.6064385175704956, 1.3831061124801636, 0.6015265583992004, 1.2646514177322388, 0.42575424909591675, 0.31847068667411804, 0.7193829417228699, 0.5860495567321777, 0.9717243909835815, 0, 0.562576174736023, 1.0423691272735596, 0.8410823941230774, 0.2970902621746063, 0.9815489053726196, 1.0513837337493896, 0.9318822622299194, 0.6688232421875, 0.8800684809684753, 0.8471208810806274, 0.9958009719848633, 1.0016874074935913, 0.6478635668754578, 0.17742964625358582, 0.6601763367652893]

    # top 60%
    # san_cityscapes_logits_threshold = [1.1793453693389893, 1.4563767910003662, 1.1595406532287598, 1.0751193761825562, 1.5221409797668457, 0.5925889611244202, 0.9914445877075195, 1.066139817237854, 1.0144094228744507, 0.9977725744247437, 1.1894174814224243, 0.8892138600349426, 0.547927975654602, 1.4713550806045532, 1.8282856941223145, 1.350669264793396, 1.013079285621643, 0.7405804991722107, 0.9798299074172974]

    # top 80%
    # san_cityscapes_logits_threshold = [1.1077282428741455, 1.225560188293457, 0.9752657413482666, 0.8524237871170044, 1.1228225231170654, 0.36980271339416504, 0.8612030744552612, 0.7514761686325073, 0.9807062149047852, 0.9040149450302124, 1.0077000856399536, 0.7110130786895752, 0.3704208433628082, 1.1130671501159668, 1.330572247505188, 1.108180284500122, 0.8068251609802246, 0.5945318937301636, 0.7882791757583618]
    # san_voc_logits_threshold = [0.017378728836774826, 0.028023002669215202, 0.011712610721588135, 0.06510047614574432, 0.035105131566524506, 0.7806247472763062, 0.04719720780849457, 0.8594886660575867, 0.014460818842053413, 0.06178871542215347, 0.028976473957300186, 0.4764820635318756, 0.026428500190377235, 0.03846154361963272, 0.013614418916404247, 0.015413750894367695, 0.024575119838118553, 0.037016887217760086, 0.09919369220733643, 0.03718540817499161]

    # top 90%
    # san_cityscapes_logits_threshold = [1.0653833150863647, 1.0547130107879639, 0.8290120363235474, 0.7166965007781982, 0.8588627576828003, 0.32600870728492737, 0.6564815640449524, 0.5544048547744751, 0.8899244070053101, 0.8156185150146484, 0.8335221409797668, 0.5381917953491211, 0.3084873855113983, 0.8742980360984802, 0.9519305229187012, 0.9366381168365479, 0.6550827622413635, 0.4871916174888611, 0.6120597124099731]

    with open(f'../mmrazor/checkpoints/{dataset}_class_san_threshold_logits_{str(top_ratio)}.pkl', "wb") as output:
        pickle.dump(threshold_logits, output)