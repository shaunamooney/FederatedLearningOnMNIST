# import
from torchvision import datasets, transforms
import syft as sy

# train data loader
def getTrainDataLoader(workers, batch_size):

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

	train_mnist = datasets.MNIST('../data', train=True, 
		download=True, transform=transform)

	federated_train_loader = sy.FederatedDataLoader(
		train_mnist.federate((bob, alice)), 
		batch_size=batch_size, shuffle=True)

	return federated_train_loader

# test data loader
def getTestDataLoader(batch_size):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

	test_mnist = datasets.MNIST('../data', train=False, 
		transform=transform)

	test_loader = torch.utils.data.DataLoader(
		test_mnist, batch_size=batch_size)

	return test_loader