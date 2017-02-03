import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1+np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs
#         final_outputs = self.activation_function(final_inputs)
        
        ### Backward pass ###
        # Output error
        output_errors = targets - final_outputs
        
        # Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        hidden_grad = hidden_outputs * (1.0 - hidden_outputs)
        
        # Update the weights
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)
        self.weights_input_to_hidden += self.lr * np.dot(hidden_errors * hidden_grad, inputs.T)
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs
#         final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
def MSE(y, Y):
    return np.mean((y-Y)**2)

class P1:
    def __init__(self):
       
        return
    def __prepare_data(self):
        rides = pd.read_csv('Bike-Sharing-Dataset/hour.csv')
#         print(rides.head())
#         rides[:24*10].plot(x='dteday', y='cnt')
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        for each in dummy_fields:
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)
        
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        data = rides.drop(fields_to_drop, axis=1)
#         print(data.head())
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        
        # Store scalings in a dictionary so we can convert back later
        self.scaled_features = {}
        for each in quant_features:
            mean, std = data[each].mean(), data[each].std()
            self.scaled_features[each] = [mean, std]
            data.loc[:, each] = (data[each] - mean)/std
        
        # Save the last 21 days 
        validation = data[-21*24:]
        data = data[:-21*24]
        
        # Separate the data into features and targets
        target_fields = ['cnt', 'casual', 'registered']
        features, targets = data.drop(target_fields, axis=1), data[target_fields]
        self.val_features, self.val_targets = validation.drop(target_fields, axis=1), validation[target_fields]
        n_records = features.shape[0]
        split = np.random.choice(features.index, 
                                 size=int(n_records*0.8), 
                                 replace=False)
        self.train_features, self.train_targets = features.ix[split], targets.ix[split]
        self.test_features, self.test_targets = features.drop(split), targets.drop(split)
        self.rides = rides
        self.validation = validation
        return
    def train_netowrk(self):
        ### Set the hyperparameters here ###
        train_features, train_targets = self.train_features, self.train_targets
        test_features, test_targets = self.test_features, self.test_targets
        val_features, val_targets = self.val_features, self.val_targets
        scaled_features = self.scaled_features
        rides = self.rides
        validation = self.validation
        
        
        epochs = 1000
        learning_rate = 0.1
        hidden_nodes = 30
        output_nodes = 1
        
        N_i = train_features.shape[1]
        network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
        
        losses = {'train':[], 'test':[]}
        for e in range(epochs):
            # Go through a random batch of 128 records from the training data set
            batch = np.random.choice(train_features.index, size=128)
            for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
                network.train(record, target)
            
            if e%(epochs/10) == 0:
                # Calculate losses for the training and test sets
                train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
                test_loss = MSE(network.run(test_features), test_targets['cnt'].values)
                losses['train'].append(train_loss)
                losses['test'].append(test_loss)
                
                # Print out the losses as the network is training
                print('Training loss: {:.4f}'.format(train_loss))
                print('Test loss: {:.4f}'.format(test_loss))
                pass
        plt.plot(losses['train'], label='Training loss')
        plt.plot(losses['test'], label='Testing loss')
        plt.legend()
        
        
        #check the predictions
        fig, ax = plt.subplots(figsize=(8,4))

        mean, std = scaled_features['cnt']
        predictions = network.run(val_features)*std + mean
        ax.plot(predictions[0], label='Prediction')
        ax.plot((val_targets['cnt']*std + mean).values, label='Data')
        ax.set_xlim(right=len(predictions))
        ax.legend()
        
        dates = pd.to_datetime(rides.ix[validation.index]['dteday'])
        dates = dates.apply(lambda d: d.strftime('%b %d'))
        ax.set_xticks(np.arange(len(dates))[12::24])
        _ = ax.set_xticklabels(dates[12::24], rotation=45)
        
        return
   
    def run(self):
        self.__prepare_data()
        self.train_netowrk()
        plt.show()

        return
    

    

    
if __name__ == "__main__":   
    obj= P1()
    obj.run()