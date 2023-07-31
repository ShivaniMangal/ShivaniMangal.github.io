
export async function barChart(filterParam) {
    d3.selectAll("svg")?.remove()
    const svg = d3.select("body").append("svg")
        .attr("id", "chart")
        .attr("class", "svgClass")
        .attr("width", "2100")
        .attr("height", 800)
    let data = await d3.csv("https://raw.githubusercontent.com/ShivaniMangal/datasets/main/SofwareDeveloperIncomeExpensesperUSACity.csv");
    let xLabel = "City";
    let meanSalaries = data.map(d => parseFloat(d["Mean Software Developer Salary (unadjusted)"]));
    const maxSalary = Math.max(...meanSalaries);


    data.sort((s1, s2) =>
        parseFloat(s1["Mean Software Developer Salary (unadjusted)"]) < parseFloat(s2["Mean Software Developer Salary (unadjusted)"]) ? 1 : -1
    )

    if(filterParam) {
        console.log(filterParam)
        if(filterParam.type==="Region" && filterParam.value!=="All Regions") {
            let filteredData = data.filter((d) => d[filterParam.type] === filterParam.value);
            data = filteredData
            console.log(data)
        }
        if(filterParam.type==="Sort"){
            data.sort((s1, s2) =>
                parseFloat(s1[filterParam.value]) < parseFloat(s2[filterParam.value]) ? 1 : -1
            )
        }
    }

    const maxSoftwareSalaryDataPoint = Math.max(...data.map(d => d["Mean Software Developer Salary (unadjusted)"]))
    const minRent = Math.min(...data.map(d => d["Cost of Living Plus Rent avg"]))



    // X axis
    let xScale = d3.scaleBand()
        .range([ 100, 1500 ])
        .domain(data.map(function(d) { return d.City }))
        .padding(0.2);
    svg.append("g")
        .attr("transform", "translate(0," + 500 + ")")
        .call(d3.axisBottom(xScale))
        .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    // Add Y axis


    let yScale = d3.scaleLinear().domain([0, maxSalary]).range([400, 0])

    svg.append("g").call(d3.axisLeft(yScale)).attr("transform", "translate(0," + 2000 + ")").attr("transform", "translate(100,100)rotate(0)")

    const colorScaleSoftware = d3.scaleOrdinal(d3.schemeTableau10)
        .domain(["West","Southeast", "Northeast", "Midwest", "Southwest"])

    const colorScaleAll = d3.scaleOrdinal(d3.schemePastel2)
        .domain(["West","Southeast", "Northeast", "Midwest", "Southwest"])

    const tip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("position", "absolute");

    svg.selectAll("software")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (d) => xScale(d.City))
        .attr("y", (d) => yScale(parseFloat(d["Mean Software Developer Salary (unadjusted)"])))
        .attr("width", xScale.bandwidth())
        .attr("height", (d) => 400 - yScale(parseFloat(d["Mean Software Developer Salary (unadjusted)"])))
        // .attr("fill", "#94C3F2")
        .attr("transform", "translate(0," + 100 + ")")
        .attr("fill", (d) => colorScaleSoftware(d.Region))
        .on("mouseover", (d) =>
            tip.style("opacity", 1)
                .html(d.City + "<br/> Mean Software Developer Salary: $" + parseFloat(d["Mean Software Developer Salary (unadjusted)"]) + "<br/> Annual COL and Rent as a Percentage of Salary: " + parseInt(d["Cost of Living Plus Rent avg"]*12*100/d["Mean Software Developer Salary (unadjusted)"]) + "%")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px")
        )
        .on("mouseout", () =>
            tip.style("opacity", 0))

    svg.selectAll("all")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (d) => xScale(d.City))
        .attr("y", (d) => yScale(parseFloat(d["Cost of Living Plus Rent avg"])*12))
        .attr("width", xScale.bandwidth())
        .attr("height", (d) => 400 - yScale(parseFloat(d["Cost of Living Plus Rent avg"])*12))
        .attr("fill", "#f4cae4")
        .attr("transform", "translate(0," + 100 + ")")
        .on("mouseover", (d) =>
            tip.style("opacity", 1)
                .html(d.City + "<br/> Monthly COL and Rent: $" + d["Cost of Living Plus Rent avg"]+ "<br/> Annual COL and Rent: $" + parseInt(d["Cost of Living Plus Rent avg"]*12)
                        +"<br/> Annual COL and Rent as a Percentage of Salary: " + parseInt(d["Cost of Living Plus Rent avg"]*12*100/d["Mean Software Developer Salary (unadjusted)"]) + "%")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px")
        )
        .on("mouseout", () =>
            tip.style("opacity", 0))


    const highestSalaryAnnotation = [
        {
            note: {
                label: "Tied between San Fransisco and Oakland California at $142,101",
                title: "Highest Mean Software Engineer Salary",
                wrap: 4000,  // try something smaller to see text split in several lines
                padding: 10 ,  // More = text lower
            },
            color: ["green"],
            x: xScale(maxSoftwareSalaryDataPoint.City),
            y: yScale(parseFloat(maxSoftwareSalaryDataPoint["Mean Software Developer Salary (unadjusted)"])),
            dy: -55,
            dx: 10
        }
    ]

    const newYorkAnnotation = [
        {
            note: {
                title: "New Jersey is easier on the Pocket than New York",
                label: "Both New York and New Jersey have the same average salary of $118,247, but the COL and Rent in New York is a staggering 53% of the salary compared to  38% of the salary in New Jersey ",
                wrap: 400,  // try something smaller to see text split in several lines
                padding: 10 ,  // More = text lower
            },
            color: ["blue"],
            x: xScale(minRent.City),
            y: yScale(parseFloat(minRent["Rent avg"])),
            dy: -130,
            dx: 2,
            subject: {
                width: -30,
                height: 90
            }
        }
    ]

    // Add annotation to the chart
    const highestSalary = d3.annotation()
        .type(d3.annotationCalloutCircle)
        .annotations(highestSalaryAnnotation)

    const newYork = d3.annotation()
        .type(d3.annotationCalloutRect)
        .annotations(newYorkAnnotation)


    svg.append("g").call(highestSalary).attr("transform", "translate(" + 120 +"," + 110 + ")")
    svg.append("g").call(newYork).attr("transform", "translate(" + 280 +"," + 300 + ")")

    svg
        .append("text")
        .attr("class", "x-label")
        .attr("x", 600) // Adjust the position based on your plot's width
        .attr("y", 600) // Adjust the position based on your plot's height
        .style("text-anchor", "middle")
        .text(xLabel);

    svg
        .append("text")
        .attr("class", "y-label")
        .attr("x", -250) // Negative height to rotate and position it properly
        .attr("y", 30) // Adjust the position based on your plot's height
        .style("text-anchor", "middle")
        .attr("transform", "translate(0,0)rotate(-90)")
        .text("Mean Software Engineer Salaries");
}

export async function toggleBarChartVisibility(hide){
    if(hide) {
        document.getElementById("chart").style.display = "none"
    } else{
        if(document.getElementById("chart"))
            document.getElementById("chart").style.display = "block"
        await barChart()
    }
}

let nextPageButton = document.getElementById("next");
nextPageButton?.addEventListener("click", async function() {
    // await toggleBarChartVisibility(true);
    window.location='../views/rent.html';
});

await barChart()