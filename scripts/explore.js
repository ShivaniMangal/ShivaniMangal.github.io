import {barChart} from "./barChart.js";
import {scatterPlot} from "./scatterPlot.js";

let isBar = true
let scatterType
await barChart()
let selectElementType = document.getElementById("types");
console.log(selectElementType)
let selectElementSort = document.getElementById("sort");
let selectElementRegion = document.getElementById("regions");
// Add an event listener to the select element
selectElementType.addEventListener("change", ()=> {getSelectedFilterValue("types")});
selectElementSort.addEventListener("change", ()=> {getSelectedFilterValue("sort")});
selectElementRegion.addEventListener("change", ()=> {getSelectedFilterValue("regions")});
export async function getSelectedFilterValue(id) {
    let selectElement = document.getElementById(id);
    let selectedValue = selectElement.value;
    console.log('Selected value: ', selectedValue);
    if(id!== "regions") document.getElementById("regions").selectedIndex = 0;
    if(id!== "sort") document.getElementById("sort").selectedIndex = 0;
    await displayChart(selectedValue, id)
}

async function displayChart(selectedValue, id) {

    // remove common annotations
    if(id==="types" && (selectedValue==="Rent avg" || selectedValue==="Cost of Living avg" || selectedValue==="Cost of Living Plus Rent avg") ) {
        document.getElementById("sorting").style.display="none"
        await scatterPlot(selectedValue);
        isBar = false
        scatterType = selectedValue
    } else if (id==="types" && selectedValue==="City") {
        document.getElementById("sorting").style.display="block"
        await barChart()
        isBar = true
    }

    if(id==="regions" && isBar){
        document.getElementById("sorting").style.display="block"
        await barChart({type: "Region", value: selectedValue})
    } else if(id==="regions" && !isBar){
        document.getElementById("sorting").style.display="none"
        await scatterPlot(scatterType, {type: "Region", value: selectedValue})
    }  else if(id==="sort" && isBar){
        document.getElementById("sorting").style.display="block"
        await barChart({type: "Sort", value: selectedValue})
    }
}

