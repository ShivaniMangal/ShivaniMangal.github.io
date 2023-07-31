export async function scatterPlot(objectName, filterParam) {
    d3.selectAll("svg")?.remove()
    if(!objectName){
        objectName = document.getElementsByTagName("script")[2].id
    }
    let xLabel=objectName
    const svg = d3.select("body").append("svg")
        .attr("id", "chart")
        .attr("class", "svgClass")
        .attr("width", "2100")
        .attr("height", 800)
    // let xLabel = "Rent";
    let data = await d3.csv("https://raw.githubusercontent.com/ShivaniMangal/datasets/main/SofwareDeveloperIncomeExpensesperUSACity.csv");
    let softwareSalaries = Array();
    let monthlyAmount = Array();


    if(filterParam) {
        if(filterParam.value!=="All Regions") {
            let filteredData = data.filter((d) => d[filterParam.type] === filterParam.value);
            data = filteredData
        }
        console.log(xLabel)
    }

    data.forEach(d => {
        softwareSalaries.push(d["Mean Software Developer Salary (unadjusted)"]);
        monthlyAmount.push(d[objectName]);
    })


    let g1 = svg.append("g").attr("transform", "translate(50,50)")
    const xScale = d3.scaleLinear().domain([Math.max(...monthlyAmount), Math.min(...monthlyAmount)]).range([800, 0])
    const yScale = d3.scaleLinear().domain([Math.max(...softwareSalaries), Math.min(...softwareSalaries)]).range([0, 300])

    const colorScale = d3.scaleOrdinal(d3.schemeTableau10)
        .domain(["West","Southeast", "Northeast", "Midwest", "Southwest"])

    const tip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("position", "absolute");


    data.forEach(d => {
        g1.append("circle").attr("cx", xScale(parseFloat(d[objectName])))
            .attr("cy", yScale(parseFloat(d["Mean Software Developer Salary (unadjusted)"]))).attr('r', 10)
            .style("fill", colorScale(d.Region))
            .on("mouseover", () =>
                tip.style("opacity", 1)
                    .html(d.City + "<br/> Mean Software Developer Salary: $" + d["Mean Software Developer Salary (unadjusted)"] + "<br/> Monthly "+ objectName + ": $" + d[objectName])
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px")
            )
            .on("mouseout", () =>
            tip.style("opacity", 0))
    });


    const xAxis = d3.axisBottom(xScale)
    const yAxis = d3.axisLeft(yScale);
    const g2 = svg.append("g").attr("transform", "translate(50,50)").call(yAxis)
    const g3 = svg.append("g").attr("transform","translate(50,350)").call(xAxis)

    const minRent = Math.min(...data.map(d => d["Rent avg"]))
    const lowestRentAnnotation = [
        {
            note: {
                label: "Akron Ohio at $879 per month",
                title: "Lowest Rent",
                wrap: 4000,  // try something smaller to see text split in several lines
                padding: 10 ,  // More = text lower
            },
            color: ["green"],
            x: xScale(minRent.City),
            y: yScale(parseFloat(minRent["Mean Software Developer Salary (unadjusted)"])),
            dy: 55,
            dx: 70
        }
    ]
    // Add annotation to the chart
    const lowestRent = d3.annotation()
        .type(d3.annotationCalloutElbow)
        .annotations(lowestRentAnnotation)
    svg.append("g").call(lowestRent).attr("transform", "translate(" + 50 +"," + 340 + ")")

    const highestRent = [
        {
            note: {
                title: "Highest Rent",
                label: "San Fransisco at $4,163.6 per month",
                wrap: 400,  // try something smaller to see text split in several lines
                padding: 10 ,  // More = text lower
            },
            color: ["blue"],
            x: xScale(minRent.City),
            y: yScale(parseFloat(minRent["Rent avg"])),
            dy: 30,
            dx: 20,
            subject: {
                width: 30,
                height: 90
            }
        }
    ]

    // Add annotation to the chart
    const highest = d3.annotation()
        .type(d3.annotationXYThreshold)
        .annotations(highestRent)
    svg.append("g").call(highest).attr("transform", "translate(" + 850 +"," + 60 + ")")

    svg
        .append("text")
        .attr("class", "x-label")
        .attr("x", 400) // Adjust the position based on your plot's width
        .attr("y", 400) // Adjust the position based on your plot's height
        .style("text-anchor", "middle")
        .text(xLabel);

    svg
        .append("text")
        .attr("class", "y-label")
        .attr("x", -260) // Negative height to rotate and position it properly
        .attr("y", 10) // Adjust the position based on your plot's height
        // .style("text-anchor", "middle")
        .attr("transform", "translate(0,0)rotate(-90)")
        .text("Mean Software Engineer Salaries");

}

let colButton = document.getElementById("colButton");
colButton?.addEventListener("click", async function() {
    window.location="costOfLiving.html";
});

let rentButton = document.getElementById("explore");
rentButton?.addEventListener("click", async function() {
    window.location="explore.html";
});

await scatterPlot()

