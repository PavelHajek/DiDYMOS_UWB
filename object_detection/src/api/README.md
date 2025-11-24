# API

## Directory structure
```
.
├── README.md
├── Dockerfile
├── Dockerfile_dev
├── rest_server.py                              # Main entrypoint module launched by uvicorn
├── api                                         # Contains API modules
│   ├── errors                                  # Registry with errors and error codes API returns
│   ├── database                                # Database ORM model and databse services (functions with CRUD operations)
│   ├── schemas                                 # Classes with API input and output specification and validation 
│   ├── routers                                 # API enpoints
│   └── services                                # Extra layer to wrap database and worker functionality, used by endpoints
└── scripts                                     # Useful testing scripts
```
