from models import simpleFCN, simpleFCN_single
import torch.optim as optim


def model_list_sample(p_list, m_list, num_sample):
    """function to get one particular sample of model and patch list"""
    p_list_n = {
        list(p_list.keys())[num_sample -
                            1]: list(p_list.values())[num_sample - 1]
    }
    m_list_n = {
        list(m_list.keys())[num_sample -
                            1]: list(m_list.values())[num_sample - 1]
    }
    return m_list_n, p_list_n


class allTraining():
    def __init__(self, sub_image_size, trainR0, trainL0, trainR1, num_patch=None):
        # getting the sub image size
        # initializing the model
        self.sub_image_size = sub_image_size
        # self.model = simpleFCN(sub_image_size *sub_image_size*3 *2)
        self.trainR0 = trainR0
        self.trainL0 = trainL0
        self.trainR1 = trainR1
        self.image_max_dim = next(iter(self.trainL0))[0].shape[0]
        self.num_patch = int(self.image_max_dim / self.sub_image_size)**2
        print(f"Max. dimension of Image : {self.image_max_dim}")
        print(f"\nMax. no. of Models/Patches possible are {self.num_patch}")
        if num_patch != None:
            self.num_patch = num_patch
        print("\nNo. of Models/patches selected : ", self.num_patch)

    def model_init(self, model, first_dim, last_dim):
        self.model_list = {}
        for i in range(self.num_patch):
            self.model_list['model_'+str(i+1)] = model(first_dim, last_dim)
        print(f"\n{len(self.model_list)} Models initialized")
        return self.model_list

    def whole_training(self, model_list, patch_size_list, criterion, epochs, training_model, applyFunc=None, applyFuncR0=None):
        self.model_list = model_list
        data_rec = {}
        patch_list = {}
        print(f"\n{len(self.model_list)} no. of models will be trained. ")
        for p, (model_key, patch_key) in enumerate(zip(self.model_list, patch_size_list)):
            model = self.model_list[model_key]
            patch = patch_size_list[patch_key]
            patch_list[patch_key] = patch
            print(
                f"\nModel : {model_key} | Patch : {patch_key} | Patch Size : {patch}")
            print(f"Training started .... ")
            # print(model)
            # print(patch_list)
            epoch_list, loss_list = training_model(self.trainR0, self.trainL0, self.trainR1, model, patch,
                                                   applyFunc, applyFuncR0,
                                                   criterion, epochs, opt=optim.Adam(model.parameters(), lr=0.0001), )

            data_rec['M'+str(p+1)] = list((epoch_list, loss_list))
            print(f"Training no. ended .... \n\n\n")

        return self.model_list, patch_list, data_rec


class allTraining_single():
    def __init__(self, sub_image_size, trainR0, trainR1, num_patch=None):
        # getting the sub image size
        # initializing the model
        self.sub_image_size = sub_image_size
        # self.model = simpleFCN(sub_image_size *sub_image_size*3 *2)
        self.trainR0 = trainR0
        # self.trainL0 = trainL0\
        self.trainR1 = trainR1
        self.image_max_dim = next(iter(self.trainR1))[0].shape[0]
        self.num_patch = int(self.image_max_dim / self.sub_image_size)**2
        print(f"Max. dimension of Image : {self.image_max_dim}")
        print(f"\nMax. no. of Models/Patches possible are {self.num_patch}")
        if num_patch != None:
            self.num_patch = num_patch
        print("\nNo. of Models/patches selected : ", self.num_patch)

    def model_init(self, first_dim, last_dim):
        self.model_list = {}
        for i in range(self.num_patch):
            self.model_list['model_' +
                            str(i+1)] = simpleFCN_single(first_dim, last_dim)
        print(f"\n{len(self.model_list)} Models initialized")
        return self.model_list

    def whole_training(self, model_list, patch_size_list, criterion, epochs, training_model, applyFunc, applyFuncR0):
        self.model_list = model_list
        data_rec = {}
        patch_list = {}
        print(f"\n{len(self.model_list)} no. of models will be trained. ")
        for p, (model_key, patch_key) in enumerate(zip(self.model_list, patch_size_list)):
            model = self.model_list[model_key]
            patch = patch_size_list[patch_key]
            patch_list[patch_key] = patch
            print(
                f"\nModel : {model_key} | Patch : {patch_key} | Patch Size : {patch}")
            print(f"Training started .... ")
            # print(model)
            # print(patch_list)
            epoch_list, loss_list = training_model(self.trainR0, self.trainR1, model, patch,
                                                   applyFunc, applyFuncR0,
                                                   criterion, epochs, opt=optim.Adam(model.parameters(), lr=0.00001), )

            data_rec['M'+str(p+1)] = list((epoch_list, loss_list))
            print(f"Training no. ended .... \n\n\n")

        return self.model_list, patch_list, data_rec
