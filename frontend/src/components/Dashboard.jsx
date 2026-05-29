import { useEffect, useState } from "react";

import axios from "axios";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer
} from "recharts";

function Dashboard() {

    const [analytics, setAnalytics] = useState({

        current_vehicle_count: 0,
        total_vehicles_seen: 0,
        traffic_density: "LOW",
        vehicles_in: 0,
        vehicles_out: 0
    });

    const [chartData, setChartData] = useState([]);

    useEffect(() => {

        const fetchAnalytics = async () => {

            try {

                const response =
                    await axios.get(
                        "http://localhost:8000/analytics"
                    );

                setAnalytics(response.data);

                setChartData((prevData) => [

                    ...prevData.slice(-20),

                    {
                        time:
                            new Date()
                            .toLocaleTimeString(),

                        vehicles:
                            response.data
                            .current_vehicle_count
                    }
                ]);

            } catch (error) {

                console.log(error);
            }
        };

        fetchAnalytics();

        const interval =
            setInterval(fetchAnalytics, 2000);

        return () =>
            clearInterval(interval);

    }, []);

    return (

        <div className="min-h-screen bg-gray-950 text-white">

            {/* HEADER */}

            <div className="p-6 border-b border-gray-800">

                <h1 className="text-4xl font-bold">
                    AI Traffic Intelligence
                </h1>

                <p className="text-gray-400 mt-2">
                    Real-Time Traffic Monitoring Dashboard
                </p>

            </div>

            {/* MAIN GRID */}

            <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* LEFT SIDE */}

                <div className="lg:col-span-2 space-y-6">

                    {/* VIDEO */}

                    <div className="bg-gray-900 rounded-2xl p-4 shadow-lg">

                        <h2 className="text-2xl font-semibold mb-4">
                            Live Traffic Feed
                        </h2>

                        <img
                            src="http://localhost:8000/video_feed"
                            alt="Live Stream"
                            className="rounded-xl border border-gray-700"
                        />

                    </div>

                    {/* CHART */}

                    <div className="bg-gray-900 rounded-2xl p-4 shadow-lg h-[400px]">

                        <h2 className="text-2xl font-semibold mb-4">
                            Vehicle Count Analytics
                        </h2>

                        <ResponsiveContainer
                            width="100%"
                            height="85%"
                        >

                            <LineChart data={chartData}>

                                <CartesianGrid strokeDasharray="3 3" />

                                <XAxis dataKey="time" />

                                <YAxis />

                                <Tooltip />

                                <Line
                                    type="monotone"
                                    dataKey="vehicles"
                                    stroke="#00FFAA"
                                    strokeWidth={3}
                                />

                            </LineChart>

                        </ResponsiveContainer>

                    </div>

                </div>

                {/* RIGHT SIDE */}

                <div className="space-y-6">

                    <DashboardCard
                        title="Vehicles"
                        value={
                            analytics.current_vehicle_count
                        }
                    />

                    <DashboardCard
                        title="Total Seen"
                        value={
                            analytics.total_vehicles_seen
                        }
                    />

                    <DashboardCard
                        title="Vehicles In"
                        value={
                            analytics.vehicles_in
                        }
                    />

                    <DashboardCard
                        title="Vehicles Out"
                        value={
                            analytics.vehicles_out
                        }
                    />

                    <DashboardCard
                        title="Traffic Density"
                        value={
                            analytics.traffic_density
                        }
                    />

                </div>

            </div>

        </div>
    );
}

function DashboardCard({ title, value }) {

    return (

        <div className="bg-gray-900 rounded-2xl p-6 shadow-lg">

            <h2 className="text-gray-400 text-lg">
                {title}
            </h2>

            <h1 className="text-4xl font-bold mt-2">
                {value}
            </h1>

        </div>
    );
}

export default Dashboard;