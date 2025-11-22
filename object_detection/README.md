# DIDYMOS API

## Run Application
The application is orchestrated using `docker-compose` - a tool for defining and running multi-container Docker application.

The application is divided into 4 separated containers:
1. Database - Postgresql DB to save predictions.
2. Tracking Worker - To inference videos and produce predictions. 
3. Broker - RabbitMQ queue for job communication between workers and API.
4. API - FastAPI backend to provide endpoints to post videos and get predictions.

It is required to have `docker` and `docker-compose` plugin installed on the target machine.

To utilize GPU in the worker container on a Jetson computer, it is required to set docker default runtime to `nvidia`.
Use this [Guide](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) for setup.
Also, beware of preinstalled drivers and packages specific for NVIDIA Jetson (e.g., do **NOT** install NVIDIA Container Toolkit).

FastAPI automatically generates docs:
* Automatic Interactive Docs (Swagger UI): http://_hostname_:_port_/docs
* Automatic Alternative Docs (ReDoc): http://_hostname_:_port_/redoc

### Preparation
Set variables such as `host_name`,`port`, `batch size`, or `timeout` in the `src/api/variables.env` based on the target machine.
The app can be further configured in the `src/api/api/config.py` file.

In the `src/api/scripts` are Postman collection with examples and useful test scripts.

### Development
Run the following commands to build and start the application in the development mode.
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

### Production
Run the following commands to build and start the application in the production mode.
```bash
docker compose up -d --build
```


## Contributing (Pull Request)

Follow these steps to add new feature to the repository:
1. Create a new branch with name `[YOUR_NAME]/[SHORT_FEATURE_DESCRIPTION]`.
2. Add your changes to the branch.
3. Run quality checks using `black`, `isort`, `flake8`:
   ```bash
   black --check .  # checks code formatting according to PEP 8 standard
   isort --check .  # checks if the package imports are sorted alphabetically and separated into sections by type
   flake8 .  # checks if the code is valid
   ```
   Code can be formated automatically using:
   ```bash
   black .
   isort .
   ```
   Make sure all the checks pass, otherwise Pull Request won't be approved.
4. Create a Pull Request to branch `dev`.
