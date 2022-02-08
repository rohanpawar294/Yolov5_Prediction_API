# Yolov5_Prediction_API

#Steps to bulid Dockerfile

git clone https://github.com/rohanpawar294/Yolov5_Prediction_API.git

cd Yolov5_Prediction_API 

docker build -t rohanpawar294/yolo_v5_api_flask:yolov5_v2 .

# Command to download Docker image 

docker pull rohanpawar294/yolo_v5_api_flask:yolov5_v2



# Run manually docker Image

docker run --device=/dev/video0:/dev/video0 -p 0.0.0.0:9999:9999 -p 8888:8888 rohanpawar294/yolo_v5_api_flask:yolov5_v2




# Commands Deployment On Kubernetes

minikube start     
                    
kubectl apply -f deplyment_apt.yaml

kubectl expose deployment kubermatic-dl-deployment  --type=LoadBalancer --target-port 5000

kubectl get pod

kubectl describe pod <pod_name>

kubectl port-forward <pod_name> 9999:9999


