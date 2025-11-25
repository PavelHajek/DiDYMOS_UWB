/*
    Demo application for visualisation of vehicle locations for the project TACR CK03000179 "Dynamic digital street model for the usage of autonomous mobility in Pilsen"
    (c) Department of Geomatics, University of West Bohemia in Pilsen, 2024
    BSD 3-Clause License
    Icons by <a href="https://www.svgrepo.com" target="_blank">SVG Repo</a>
 */
var czmlHeaderCamera = {
    id: "document",
    name: "CZML Camera objects",
    version: "1.0",
    clock: {
        multiplier: 10
    }
};

function person(id, name, avail, epoch, loc) {
    this.id = id;
    this.name = name;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.billboard = {};
    this.billboard.image = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgZmlsbD0iI2RmZDkyMCI+CiAgPGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiLz4KICA8ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KICA8ZyBpZD0iU1ZHUmVwb19pY29uQ2FycmllciI+CiAgICA8cGF0aCBkPSJNMTMuOSAyLjk5OUExLjkgMS45IDAgMSAxIDEyIDEuMWExLjkgMS45IDAgMCAxIDEuOSAxLjg5OXpNMTMuNTQ0IDZoLTMuMDg4YTEuODU1IDEuODU1IDAgMCAwLTEuOCAxLjQwNWwtMS42NjIgNi42NTJhLjY2Ny42NjcgMCAwIDAgLjE0LjU3My44NzMuODczIDAgMCAwIC42NjUuMzMuNzE4LjcxOCAwIDAgMCAuNjUzLS40NDVMMTAgOS4xVjEzbC0uOTIyIDkuMjE5YS43MS43MSAwIDAgMCAuNzA3Ljc4MWguMDc0YS42OS42OSAwIDAgMCAuNjc4LS41NjNMMTIgMTQuNTgzbDEuNDYzIDcuODU0YS42OS42OSAwIDAgMCAuNjc4LjU2M2guMDc0YS43MS43MSAwIDAgMCAuNzA3LS43ODFMMTQgMTNWOS4xbDEuNTQ4IDUuNDE1YS43MTguNzE4IDAgMCAwIC42NTMuNDQ0Ljg3My44NzMgMCAwIDAgLjY2NS0uMzI5LjY2Ny42NjcgMCAwIDAgLjE0LS41NzNsLTEuNjYyLTYuNjUyQTEuODU1IDEuODU1IDAgMCAwIDEzLjU0NCA2eiIvPgogICAgPHBhdGggZmlsbD0ibm9uZSIgZD0iTTAgMGgyNHYyNEgweiIvPgogIDwvZz4KPC9zdmc+Cg==';
}

function car(id, name, avail, epoch, loc) {
    this.id = id;
    this.name = name;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.billboard = {};
    this.billboard.image = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBMaWNlbnNlOiBNSVQuIE1hZGUgYnkgYml0Y29pbmRlc2lnbjogaHR0cHM6Ly9naXRodWIuY29tL2JpdGNvaW5kZXNpZ24vYml0Y29pbi1pY29ucyAtLT4KPHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xOC45MiA1LjAxQzE4LjcyIDQuNDIgMTguMTYgNCAxNy41IDRINi41QzUuODQgNCA1LjI5IDQuNDIgNS4wOCA1LjAxTDMgMTFWMTlDMyAxOS41NSAzLjQ1IDIwIDQgMjBINUM1LjU1IDIwIDYgMTkuNTUgNiAxOVYxOEgxOFYxOUMxOCAxOS41NSAxOC40NSAyMCAxOSAyMEgyMEMyMC41NSAyMCAyMSAxOS41NSAyMSAxOVYxMUwxOC45MiA1LjAxWk02LjUgMTVDNS42NyAxNSA1IDE0LjMzIDUgMTMuNUM1IDEyLjY3IDUuNjcgMTIgNi41IDEyQzcuMzMgMTIgOCAxMi42NyA4IDEzLjVDOCAxNC4zMyA3LjMzIDE1IDYuNSAxNVpNMTcuNSAxNUMxNi42NyAxNSAxNiAxNC4zMyAxNiAxMy41QzE2IDEyLjY3IDE2LjY3IDEyIDE3LjUgMTJDMTguMzMgMTIgMTkgMTIuNjcgMTkgMTMuNUMxOSAxNC4zMyAxOC4zMyAxNSAxNy41IDE1Wk01IDEwTDYuNSA1LjVIMTcuNUwxOSAxMEg1WiIgZmlsbD0iIzMzY2NmZiIvPgo8L3N2Zz4=';
}

function motorcycle(id, name, avail, epoch, loc) {
    this.id = id;
    this.name = name;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.billboard = {};
    this.billboard.image = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLUxpY2Vuc2U6IE1JVC4gTWFkZSBieSBLZW5hbiBHdW5kb2dhbjogaHR0cHM6Ly9naXRodWIuY29tL2tlbmFuZ3VuZG9nYW4vZm9udGlzdG8tLT4KPHN2ZyBmaWxsPSIjMDBmZjk5IiB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Qm94PSItMy41IDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPHBhdGggZD0ibTguNjMyIDE1LjUyNmMtMS4xNjIuMDAzLTIuMTAyLjk0NC0yLjEwNiAyLjEwNXY0LjI2NC4wNDFjMCAxLjE2My45NDMgMi4xMDYgMi4xMDYgMi4xMDZzMi4xMDYtLjk0MyAyLjEwNi0yLjEwNmMwLS4wMTQgMC0uMDI5IDAtLjA0M3YuMDAyLTQuMjYzYy0uMDAzLTEuMTYxLS45NDQtMi4xMDItMi4xMDQtMi4xMDZ6Ii8+CiAgPHBhdGggZD0ibTE2LjI2MyAyLjYzMWgtNC4wNTNjLS40OTEtMS41MzctMS45MDctMi42MzEtMy41NzktMi42MzFzLTMuMDg3IDEuMDk0LTMuNTcxIDIuNjA0bC0uMDA3LjAyN2gtNGMtLjU4MSAwLTEuMDUzLjQ3MS0xLjA1MyAxLjA1M3MuNDcxIDEuMDUzIDEuMDUzIDEuMDUzaDQuMDUzYy4yNjguODk5Ljg1IDEuNjM1IDEuNjE1IDIuMDk2bC4wMTYuMDA5Yy0yLjg3MS44NjctNC45MjkgMy40OC00Ljk0NyA2LjU3N3Y1LjUyOGMuMDA5Ljk1Ni43ODEgMS43MjggMS43MzYgMS43MzdoMS40MjJ2LTNjMC0yLjA2NCAxLjY3My0zLjczNyAzLjczNy0zLjczN3MzLjczNyAxLjY3MyAzLjczNyAzLjczN3YzaDEuNDIxYy45NTctLjAwOCAxLjczLS43ODEgMS43MzgtMS43Mzd2LTUuNDc0Yy0uMDAxLTMuMTA1LTIuMDY3LTUuNzI2LTQuODk5LTYuNTY3bC0uMDQ4LS4wMTJjLjc4Mi0uNDcxIDEuMzYzLTEuMjA2IDEuNjI1LTIuMDhsLjAwNy0uMDI2aDQuMDUzYy41ODEtLjAwMiAxLjA1MS0uNDcyIDEuMDUzLTEuMDUzLS4wMjMtLjYwMS0uNTA1LTEuMDgzLTEuMTA0LTEuMTA1aC0uMDAyem0tNy42MzIgMy4yMDljLTEuMTYzIDAtMi4xMDYtLjk0My0yLjEwNi0yLjEwNnMuOTQzLTIuMTA2IDIuMTA2LTIuMTA2IDIuMTA2Ljk0MyAyLjEwNiAyLjEwNmMuMDAxLjAxOC4wMDEuMDM5LjAwMS4wNiAwIDEuMTMtLjkxNiAyLjA0Ni0yLjA0NiAyLjA0Ni0uMDIxIDAtLjA0MiAwLS4wNjMtLjAwMWguMDAzeiIvPgo8L3N2Zz4K';
}

function bicycle(id, name, avail, epoch, loc) {
    this.id = id;
    this.name = name;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.billboard = {};
    this.billboard.image = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiA/Pgo8IS0tTGljZW5zZTogUEQuIE1hZGUgYnkgTGF5bGFSZW46IGh0dHBzOi8vZHJpYmJibGUuY29tL0xheWxhUmVuLS0+Cjxzdmcgd2lkdGg9IjI1cHgiIGhlaWdodD0iMjVweCIgdmlld0JveD0iMCAwIDE2IDE2IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik0xMS41IDNDMTIuMzI4NCAzIDEzIDIuMzI4NDMgMTMgMS41QzEzIDAuNjcxNTczIDEyLjMyODQgMCAxMS41IDBDMTAuNjcxNiAwIDEwIDAuNjcxNTczIDEwIDEuNUMxMCAyLjMyODQzIDEwLjY3MTYgMyAxMS41IDNaIiBmaWxsPSIjMDMwNzA4Ii8+CiAgPHBhdGggZD0iTTAgMTJDMCAxMC40Njk3IDEuMTQ1NzggOS4yMDcwMiAyLjYyNjI2IDkuMDIzMDRMNS4wMDAwMSAxMC42MDU1VjE0LjIzNjFDNC40NjkyNSAxNC43MTExIDMuNzY4MzYgMTUgMyAxNUMxLjM0MzE1IDE1IDAgMTMuNjU2OCAwIDEyWiIgZmlsbD0iIzAzMDcwOCIvPgogIDxwYXRoIGQ9Ik0xMSAxNC4yMzYxQzExLjUzMDggMTQuNzExMSAxMi4yMzE2IDE1IDEzIDE1QzE0LjY1NjkgMTUgMTYgMTMuNjU2OSAxNiAxMkMxNiAxMC4zNDMxIDE0LjY1NjkgOSAxMyA5QzEyLjIzMTYgOSAxMS41MzA4IDkuMjg4ODUgMTEgOS43NjM4OVYxNC4yMzYxWiIgZmlsbD0iIzAzMDcwOCIvPgogIDxwYXRoIGQ9Ik04LjI4MTExIDUuMDE0NDRDOC4yOTUyNCA1LjAwNTAzIDguMzExODQgNSA4LjMyODgxIDVDOC4zNjEzOSA1IDguMzkxMTcgNS4wMTg0IDguNDA1NzQgNS4wNDc1NEw5LjM4MTk3IDdIMTNWNUgxMC42MThMMTAuMTk0NiA0LjE1MzExQzkuODQxMjQgMy40NDY0MSA5LjExODkzIDMgOC4zMjg4MSAzQzcuOTE2OTkgMyA3LjUxNDM3IDMuMTIxOSA3LjE3MTcxIDMuMzUwMzRMNC44NjEzMiA0Ljg5MDZDNC4zMjMyMiA1LjI0OTM0IDQgNS44NTMyNyA0IDYuNUM0IDcuMTQ2NzMgNC4zMjMyMiA3Ljc1MDY2IDQuODYxMzIgOC4xMDk0TDcgOS41MzUxOFYxM0g5VjguNDY0ODJMNi4wNTI3OCA2LjVMOC4yODExMSA1LjAxNDQ0WiIgZmlsbD0iI2ZmZmYwMCIvPgo8L3N2Zz4K';
}

/* 
    Function parses camera object from JSON response
*/
async function getCameraObjectJSON(file) {
    
	let response = await fetch(file);
	let respJSON = await response.json();
    
	let startDate = respJSON.start_time;
	let endDate = respJSON.end_time;
	let startD = Date.parse(startDate);
	let endD = Date.parse(endDate);
	
    // number of frames
    let framesLen = respJSON.frames.length;
    //framesLen = 1;
    
    // objects
    let camObjMap = new Map();
  // ['car', 'bicycle', 'person', 'motorcycle']
    let carMap = new Map();
    let bicycleMap = new Map();
    let personMap = new Map();
    let motorcycleMap = new Map();  
    
    for (let i = 0; i < framesLen; i++) {
        let frame = respJSON.frames[i];
        let camera_obj = frame.camera_objects;
        for(let j = 0; j < camera_obj.length; j++){

            let cam_object = camera_obj[j];      
            let co_class = cam_object.class_name;
            let co_trackId = cam_object.track_id;
            let timest = cam_object.timestamp+"Z";
            let position = cam_object.position;
            let xpos = position.x;
            let ypos = position.y;
            
            let locObject = [];
            let currTime = Date.parse(timest);
            let timeDiff = (currTime - startD) / 1000;
            locObject.push(timeDiff);
			locObject.push(Number.parseFloat(xpos));
			locObject.push(Number.parseFloat(ypos));
			locObject.push(0);
            
            switch (co_class) {
                case 'car':
                    if(carMap.has(co_trackId)){
                        let locations = carMap.get(co_trackId);
                        locations = locations.concat(locObject); 
                        carMap.set(co_trackId, locations);
                    }
                    else{
                        carMap.set(co_trackId, locObject);
                    }
                        break;
                    case 'person':
                        if(personMap.has(co_trackId)){
                            let locations = personMap.get(co_trackId);
                            locations = locations.concat(locObject); 
                            personMap.set(co_trackId, locations);
                        }
                        else{
                            personMap.set(co_trackId, locObject);
                        }
                        break;
                    case 'bicycle':
                        if(bicycleMap.has(co_trackId)){
                            let locations = bicycleMap.get(co_trackId);
                            locations = locations.concat(locObject); 
                            bicycleMap.set(co_trackId, locations);
                        }
                        else{
                            bicycleMap.set(co_trackId, locObject);
                        }
                        break;
                    case 'motorcycle':
                        if(motorcycleMap.has(co_trackId)){
                            let locations = motorcycleMap.get(co_trackId);
                            locations = locations.concat(locObject); 
                            motorcycleMap.set(co_trackId, locations);
                        }
                        else{
                            motorcycleMap.set(co_trackId, locObject);
                        }
                        break;
                    default:
                        if (!camObjMap.has(co_class)){
                            let tracks = [];
                            tracks.push(co_trackId);
                            camObjMap.set(co_class, tracks);
                        }
                        else{
                            let camObj = camObjMap.get(co_class);
                            camObj.push(co_trackId);
                            camObjMap.set(co_class, camObj);
                        }
                    }
                }
            }
    
	let czml = [];
	czmlHeaderCamera.clock.interval = startDate + "/" + endDate;
	czmlHeaderCamera.clock.currentTime = startDate;
	czml.push(czmlHeaderCamera);
    
    for (const x of personMap.keys()) {
            const personCur = new person(    
                        x,
                        'Person: '+ x,
                        startDate + "/" + endDate,
                        startDate,
                        personMap.get(x));
            czml.push(personCur); 
    }
    for (const y of carMap.keys()) {
            const carCur = new car(    
                        y,
                        'Car: '+ y,
                        startDate + "/" + endDate,
                        startDate,
                        carMap.get(y));
            czml.push(carCur); 
    }
    for (const z of bicycleMap.keys()) {
            const bicyCur = new bicycle(    
                        z,
                        'Bicycle: '+ z,
                        startDate + "/" + endDate,
                        startDate,
                        bicycleMap.get(z));
            czml.push(bicyCur); 
    }
    for (const a of motorcycleMap.keys()) {
            const motoCur = new motorcycle(    
                        a,
                        'Car: '+ a,
                        startDate + "/" + endDate,
                        startDate,
                        motorcycleMap.get(a));
            czml.push(motoCur); 
    }
	return czml;
}