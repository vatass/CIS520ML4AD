import numpy as np 
import argparse
from kdemodels import Convolutional_LongitudinalModeling, KernelDensityLayer, ValidationNet


def build_network(): 
    # descriptor_net = LSTM_LongitudinalModeling(latent_dim=latent_dim)
    descriptor_net = Convolutional_LongitudinalModeling()
    kd_layer = KernelDensityLayer(bandwidth=bandwidth)

    # network: Union[ValidationNet, nn.DataParallel]
    network = ValidationNet(descriptor_net, kd_layer)

def train_network(): 
    
    # Initialize KDE 
    initialKDE = [] 
    with torch.nograd(): 

        for sample in train_loader: 

            loss, vector = network(sample)
            
            initialKDE.append(vector)
        
        KDEinit = torch.cat(initialKDE)
            

    network.KernelDensityLayer.set_kdpoints(KDEinit)

    network.zero_grad()
    for i in range(epochs): 

        for sample in train_loader: 

            output = network(sample)

            loss, vector = output


            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 


        # update KDE vectors
        KDEupdate = []
        with torch.no_grad(): 

            for sample in train_loader: 

                output = network(sample)
                loss, vector = output 

                KDEupdate.append(vector)


            # write the vector 
            KDEtensor = torch.cat(KDEupdate)
            network.KernelDensityLayer.set_points(KDEupdate)

        # validation 
        if epoch % validate_every == 0 and epoch!=0: 
            
            with torch.no_grad(): 

                for sample in test_loader: 

                    output = network(sample)
                
                    loss, vector = output 

                    # do smth here !

def train(name, 
          model_params,
          training_params,
          output_path=None):

    ### DATA LOAD ### 
    base ='/home/vtass/Desktop/UPenn-1st-Year/FALL-2020/CIS520/Project'
    savepath = [base + '/traincn.pkl', base + '/testcn.pkl', base + '/testood.pkl']
    train_dataset = ADNI_Loader(source_data_filename='/home/vtass/Desktop/UPenn-1st-Year/FALL-2020/CIS520/Project/longitudinal_dataset.pkl', savepath=savepath, train_diagnosis='CN->CN') 
    test_dataset = ADNI_TestLoader(path=savepath[1])


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4)

    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=4)

    print('Train set', len(train_dataset))
    print('Test set', len(test_dataset))
 
   
    #### 

    LOGGER.info("Building network...")
    network = build_network(**model_params)   # defines the network, latent dimension and bandwith of gaussian 
    
    optimizer = torch.optim.Adam(
    network.parameters(),
    lr=training_params["learning_rate"],
    weight_decay=training_params["weight_decay"])

    print('Network to be used', network)
    network = network.to(device)

    print("Training Procedure Starts!")
    train_network(output_path, network, train_dataset, test_dataset, training_params)


    print("Save the trained network")





    ## Get the Embeddings for Longitudinal ADNI dataset
    # get the network embeddings from the whole dataset
    # Load the ADNI dataset and then fw pass the model 
    with open('../../longitudinal_dataset.pkl', 'rb') as f:
        d = pickle.load(f)    

    embeddings_convo_classif = [] 

    d = d['dataset']
    for feature,label in d: 
        feature = torch.tensor(feature)
        feature = feature.float() 
        with torch.no_grad():
            feature = np.expand_dims(np.expand_dims(feature,0),0) 
            feature = torch.from_numpy(feature).to(device)
            embeddingm_ = network(feature,True)
            print('embedding', embeddingm_.shape)
            embeddings_convo_classif.append((embeddingm_, label))

    emb_d = {'dataset': embeddings_convo_classif}
    with open('../../long_embeddings_kde.pickle', 'wb') as f:
        pickle.dump(emb_d, f)




if __name__ == "__main__": 


    parser = argparse.ArgumentParser()
    parser.add_argument('name',  metavar='N', type=int, nargs='+',
                    help='Set a name for the running experiment')
    parser.add_argument('--output_path', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='Set the output path for the model and the results to be saved')

    parser.add_argument('--train_dataset', type=str, help='Add the full path to train data')
    parser.add_argument('--validation_dataset', type=str, help='Add the full path to validation data')
    parser.add_argument('--learning_rate', type=float, help='Learning Rate for training')
    parser.add_argument('--weight_decay', type=float, help='Weight Decay for the optimizer') 
    parser.add_argument('--validate_every', type=int, help='Perform validation every #validate every epochs')
    parser.add_argument('--update_every', type=int, help='Perform KDE update every #update_every epochs')
    

    args = parser.parse_args()
    print(args.accumulate(args.integers))

    model_params = {'latent_dimension': args['latent_dimension'], 'bandwidth': args['bandwidth'] }
    training_params = {'epochs': args['epochs'], 'validate_every': args['validate_every'] , 'learning_rate': args['learning_rate'], 'update_every': args['update_every']} 


    train(name=args['name'], 
    model_params=model_params, 
    training_params=training_params, 
    output_path=args['output_path'])





