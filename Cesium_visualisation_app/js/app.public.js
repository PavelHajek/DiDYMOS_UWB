/*
    Demo application for visualisation of vehicle locations for the project TACR CK03000179 "Dynamic digital street model for the usage of autonomous mobility in Pilsen"
    (c) Department of Geomatics, University of West Bohemia in Pilsen, 2024
    BSD 3-Clause License
    Icons by <a href="https://www.svgrepo.com" target="_blank">SVG Repo</a>
 */

var czmlHeader = {
    id: "document",
    name: "CZML Trams",
    version: "1.0",
    clock:{
        multiplier: 10,
    },
  };

function vehicle(id, avail, epoch, loc){
    this.id = id;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.point = {};
    this.point.color={};
    this.point.color.rgba = [255, 255, 255, 128];
    
    this.point.outlineColor= {};
    this.point.outlineColor.rgba = [255, 255, 0, 255];
    
    this.point.outlineWidth = 3;
    this.point.pixelSize = 15;   
}

function vehicle2(id, avail, epoch, loc){
    this.id = id;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;

    this.billboard = {};
    this.billboard.image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAsQAAALEBxi1JjQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAUUSURBVFiF7ZdrbBRVFMd/M3Pn0deyLdBS2m7ZthgKSIO8G1BExRSjGIIYQhCMQBRNJAZEv9XEiMHaL0RMCCEqhqhEFAspQgxgl0eMyisEUuhjl0JbW1rabsu+rx/aDhSBLQkSE/knN7n3nP85959zz8ydgYcYHEYKIQ6YpukxTdPjdDrLS0s35GVmZq3JzMxaU1q6Ic/pdJb3+4UQB4CRg0msAMuBjDi8PGBV8awnAMhxjWL0mEIar1wGIHNkFhfOn+OSrx6Ao1WHAbYAtXHyNgtgNpAVh5gK4M4fbRvqai4OmAtdt/19Aib1Cb8bLsfx2xirKEoIkIMZfdyxg01+L1gO1N3FX9fHGTTETfMswIzDH94Xc6fSij5OvNIHuan8Raah+Bhkee/XME3FC0xQEkzlZMnslPEfvTNC04USR/j9QTgief/Tpui+w11nFEUheuALt/rUjOQHsnk/fjnm55nldVEhJWrl4S5qfKEHKqC6LoiUaAog09IcYcsyYv3OcCiitrRe0zNHpIUUVZX99u7ugOb392gZGWkhgGvtXQLAmZoSAWhubjOSkxOjSUlWtD9GxmJKY1ObMXyYM6wbwt4jEAipbW2dOoDcsXuvrG3tsMeeQx4JyJM1vgH2jZs2y2yXy14vXLxELly8xF5nu1xy46bNA2JO1vgkIPcc8gyw79i9VwJS/XcLHR8PBQiA0vVrQykOh90gPd3dKmAsW7QgIDTNJre2tGgyJgW9t+g/IGNSflZeFvl2+5d2E0aiUQBr3ZuvhxKTkuw9ujo7VcAQALphGKZp2YlM06Lv6r1hBHRdx9/tDwHG7QQEQsFwamKiYZqWbueC/lwDYgJG4EYFHps8jVx3vNc3nDrxB0c9h+/KKRw/gaKJk+Lm8tbVcvb0qf9ID2zftmXQAekj0u8oWlNRK3btpGLXznsTUPZeJkVjrHhc9nv8fL1XCp+3HgC/vwuA/rWmSLFuxXDmzox/r5w6H2Dtx429AiaNS2D2tKS4QS1tET7ZeonZk4oG2PdV/GTPJ47N4eni+AKE1vsg3fxBgr8nxvnaII+MMnEkq7S2R/BeDjNutIVlKsyZnoyqKjy/YBHZrtwBCRt8Xip2fcec6b2bB4KSsxcCjMo2GOrU6PTHqK4PMibPJDnxxinaAo6f7GHeSh/tHWGSEgVL5w9h6852IpEYOZkWB7fnku8ymDU5hSsNPh4tmjhAwG9Hq5g1OYWMYYKL3hBPvuKloTGAECorXkrlqx876LkeIXWITuVWlx2nAaXDUjXKtrWTmlHIshVvEA7H+GHvOZ59bj4vLnyZ6upaDh5tIhyRNDSF+PN0M1NmzEJRessopWRfxU4K3Sqd/hgfbm6hO5LBa6vfJsXh5JvvTzC1+HEWL32Vlpar7Kr0ImWMYyd6EJahHP98x9XcQIj0khfyNcM0yc3L58ivh3AXjMZKSGBktpsjnvrg72e62wCuB+WIS756xZXrBuCSr56Ojm6536M07fd0EQyTNmPmVNOyEnDnFwAwKq8AwzTJyc2jsuJU1Ntw/S/LULx2KVRVLTctKzi1eKZMcTiCQHRoenpwyvRiqWkiDKy2z02IKndeQWzO3BI5Z26JdOcXxIQQVTedyFuaJsJTphfLoenpQSCa4nAEpxbPlKZlBVVVLbttYwLrhRA/A6X0fiWXa0JUAitv4c7Tdb2q/1dM1/UqoOQWzipNiH1AOb2/aR/05X6XW5r//42/ARfmJ4pU43w/AAAAAElFTkSuQmCC';
}

function specVehicle(id, avail, epoch, loc){
    this.id = id;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.point = {};
    this.point.color={};
    this.point.color.rgba = [0, 204, 255, 128];
    
    this.point.outlineColor= {};
    this.point.outlineColor.rgba = [255, 255, 0, 255];
    
    this.point.outlineWidth = 3;
    this.point.pixelSize = 15;   
}

function testDevice(id, avail, epoch, loc){
    this.id = id;
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.point = {};
    this.point.color={};
    this.point.color.rgba = [255, 51, 204, 128];
    
    this.point.outlineColor= {};
    this.point.outlineColor.rgba = [255, 0, 0, 128];
    
    this.point.outlineWidth = 3;
    this.point.pixelSize = 15;
}

function testVehicle(id, avail, epoch, loc){
    this.id = id;
    this.name = "Test tram";
    this.description = 'Test tram of the project <a href="https://starfos.tacr.cz/cs/projekty/CK03000179" target="_blank">DiDYMOS</a>';
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.point = {};
    this.point.color={};
    this.point.color.rgba = [255, 255, 0, 128];
    
    this.point.outlineColor= {};
    this.point.outlineColor.rgba = [255, 0, 0, 128];
    
    this.point.outlineWidth = 5;
    this.point.pixelSize = 15;
}

function testVehicle2(id, avail, epoch, loc){
    this.id = id;
    this.name = "Test tram";
    this.description = 'Test tram of the project <a href="https://starfos.tacr.cz/cs/projekty/CK03000179" target="_blank">DiDYMOS</a>';
    this.availability = avail;
    this.position = {};
    this.position.epoch = epoch;
    this.position.cartographicDegrees = loc;
    
    this.billboard = {};
    this.billboard.image = 
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAsQAAALEBxi1JjQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAUUSURBVFiF7ZdrbBRVFMd/M3Pn0deyLdBS2m7ZthgKSIO8G1BExRSjGIIYQhCMQBRNJAZEv9XEiMHaL0RMCCEqhqhEFAspQgxgl0eMyisEUuhjl0JbW1rabsu+rx/aDhSBLQkSE/knN7n3nP85959zz8ydgYcYHEYKIQ6YpukxTdPjdDrLS0s35GVmZq3JzMxaU1q6Ic/pdJb3+4UQB4CRg0msAMuBjDi8PGBV8awnAMhxjWL0mEIar1wGIHNkFhfOn+OSrx6Ao1WHAbYAtXHyNgtgNpAVh5gK4M4fbRvqai4OmAtdt/19Aib1Cb8bLsfx2xirKEoIkIMZfdyxg01+L1gO1N3FX9fHGTTETfMswIzDH94Xc6fSij5OvNIHuan8Raah+Bhkee/XME3FC0xQEkzlZMnslPEfvTNC04USR/j9QTgief/Tpui+w11nFEUheuALt/rUjOQHsnk/fjnm55nldVEhJWrl4S5qfKEHKqC6LoiUaAog09IcYcsyYv3OcCiitrRe0zNHpIUUVZX99u7ugOb392gZGWkhgGvtXQLAmZoSAWhubjOSkxOjSUlWtD9GxmJKY1ObMXyYM6wbwt4jEAipbW2dOoDcsXuvrG3tsMeeQx4JyJM1vgH2jZs2y2yXy14vXLxELly8xF5nu1xy46bNA2JO1vgkIPcc8gyw79i9VwJS/XcLHR8PBQiA0vVrQykOh90gPd3dKmAsW7QgIDTNJre2tGgyJgW9t+g/IGNSflZeFvl2+5d2E0aiUQBr3ZuvhxKTkuw9ujo7VcAQALphGKZp2YlM06Lv6r1hBHRdx9/tDwHG7QQEQsFwamKiYZqWbueC/lwDYgJG4EYFHps8jVx3vNc3nDrxB0c9h+/KKRw/gaKJk+Lm8tbVcvb0qf9ID2zftmXQAekj0u8oWlNRK3btpGLXznsTUPZeJkVjrHhc9nv8fL1XCp+3HgC/vwuA/rWmSLFuxXDmzox/r5w6H2Dtx429AiaNS2D2tKS4QS1tET7ZeonZk4oG2PdV/GTPJ47N4eni+AKE1vsg3fxBgr8nxvnaII+MMnEkq7S2R/BeDjNutIVlKsyZnoyqKjy/YBHZrtwBCRt8Xip2fcec6b2bB4KSsxcCjMo2GOrU6PTHqK4PMibPJDnxxinaAo6f7GHeSh/tHWGSEgVL5w9h6852IpEYOZkWB7fnku8ymDU5hSsNPh4tmjhAwG9Hq5g1OYWMYYKL3hBPvuKloTGAECorXkrlqx876LkeIXWITuVWlx2nAaXDUjXKtrWTmlHIshVvEA7H+GHvOZ59bj4vLnyZ6upaDh5tIhyRNDSF+PN0M1NmzEJRessopWRfxU4K3Sqd/hgfbm6hO5LBa6vfJsXh5JvvTzC1+HEWL32Vlpar7Kr0ImWMYyd6EJahHP98x9XcQIj0khfyNcM0yc3L58ivh3AXjMZKSGBktpsjnvrg72e62wCuB+WIS756xZXrBuCSr56Ojm6536M07fd0EQyTNmPmVNOyEnDnFwAwKq8AwzTJyc2jsuJU1Ntw/S/LULx2KVRVLTctKzi1eKZMcTiCQHRoenpwyvRiqWkiDKy2z02IKndeQWzO3BI5Z26JdOcXxIQQVTedyFuaJsJTphfLoenpQSCa4nAEpxbPlKZlBVVVLbttYwLrhRA/A6X0fiWXa0JUAitv4c7Tdb2q/1dM1/UqoOQWzipNiH1AOb2/aR/05X6XW5r//42/ARfmJ4pU43w/AAAAAElFTkSuQmCC';
}


function vozidlo(tramId, typ, locs){
    this.objectId = tramId;
    this.type = typ;
    this.locations = locs;
}

/*
    main function to process the data JSON
*/
async function getJSON(file) {
    //console.log(file);
	let response = await fetch(file);
	let respJSON = await response.json();
	let framesLen = respJSON.frames.length;
	let startDate = respJSON.start_time;
	let endDate = respJSON.end_time;
	let startD = Date.parse(startDate);
	let endD = Date.parse(endDate);
	
    //framesLen = 1;
	let tramsMap = new Map();
	for (let i = 0; i < framesLen; i++) {
		let frame = respJSON.frames[i];
		//console.log("Frame: " + i);
        
        // process traffic_vehicles_cits Array
		let traff_veh = frame.traffic_vehicles_cits
		for (let i = 0; i < traff_veh.length; i++) {
			let tramId = traff_veh[i].objectId;
            let vehType = traff_veh[i].type;
            
            // process locations Array
			let locat = traff_veh[i].locations;
			let locTram = [];
			for (let x in locat) {
				locTram.push((Date.parse(locat[x].timestamp) - startD) / 1000);
				locTram.push(Number.parseFloat(locat[x].longitude));
				locTram.push(Number.parseFloat(locat[x].latitude));
				locTram.push(0);
			}
            
            let currTram = tramsMap.get(tramId);
			if (currTram == undefined) {
                const tram = new vozidlo(tramId, vehType, locTram);
                tramsMap.set(tramId, tram);
			} else {
                if(currTram.locations != undefined){
                    currTram.locations = currTram.locations.concat(locTram);
                    tramsMap.set(tramId, currTram);
                }
				
			}
		}
	}
    
    /* 
      create CZML object
    */
	let czml = [];
	czmlHeader.clock.interval = startDate + "/" + endDate;
	czmlHeader.clock.currentTime = startDate;
	czml.push(czmlHeader);
	// test tram CAM_100
    
    for (const x of tramsMap.values()) {
        if(x.objectId == "CAM_100"){
            const testTram = new testVehicle2(
                                    x.objectId,
                                    startDate + "/" + endDate,
                                    startDate,
                                    x.locations);
	        czml.push(testTram);
        }
        else{
            if(x.type == "specialVehicle"){
                const veh = new specVehicle(    
                                    x.objectId,
                                    startDate + "/" + endDate,
                                    startDate,
                                    x.locations);
            
	           czml.push(veh);
            }
            else if(x.type == "device"){
                const veh = new testDevice(    
                                    x.objectId,
                                    startDate + "/" + endDate,
                                    startDate,
                                    x.locations);
            
	           czml.push(veh);
            }
            else {
                const veh = new vehicle2(    
                                    x.objectId,
                                    startDate + "/" + endDate,
                                    startDate,
                                    x.locations);
            
	           czml.push(veh);                
            }
            
        }
    }

	return czml;
}