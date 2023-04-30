var map = L.map('map').setView([37.7749, -122.4194], 13);

L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
    maxZoom: 20,
    subdomains:['mt0','mt1','mt2','mt3']
}).addTo(map);

// Define variables to hold the address, latitude, and longitude
var address, latitude, longitude;

// Add a search box to the map and listen for when the user selects a place
var geocoder = L.Control.geocoder({
  defaultMarkGeocode: false
}).addTo(map);

geocoder.on('markgeocode', function(event) {
  // Get the address, latitude, and longitude from the event object
  address = event.geocode.name;
  latitude = event.geocode.center.lat;
  longitude = event.geocode.center.lng;
  
  // Set the map's view to the selected location
  map.setView([latitude, longitude], 18);
  
  // Show a popup on the map with the name of the selected location
  L.popup()
    .setLatLng([latitude, longitude])
    .setContent(address)
    .openOn(map);
  
  // Do something with the address, latitude, and longitude values
  console.log('Address: ' + address);
  console.log('Latitude: ' + latitude);
  console.log('Longitude: ' + longitude);
});

function submitAddress() {

  // Create a new XMLHttpRequest object
  var xhr = new XMLHttpRequest();

  // Set up the request
  xhr.open("POST", "/getinfo");

  // Set the request header
  xhr.setRequestHeader("Content-Type", "application/json");

  // Set up a callback function for when the request completes
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      // Handle the response from the Flask endpoint
      var response = xhr.responseText;
      console.log(response);
    }
  };

  // Create a data object to send to the Flask endpoint
  var data = JSON.stringify({ address: address, latitude: latitude, longitude: longitude });

  // Send the request with the data object as the request body
  xhr.send(data);
}
