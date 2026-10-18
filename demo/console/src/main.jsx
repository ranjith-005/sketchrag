import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "./index.css";
import Layout from "./components/Layout";
import { LiveProvider, QueryProvider } from "./lib/store";
import Overview from "./pages/Overview";
import Stream from "./pages/Stream";
import Memory from "./pages/Memory";
import Query from "./pages/Query";
import Evaluation from "./pages/Evaluation";
import Config from "./pages/Config";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <LiveProvider>
        <QueryProvider>
          <Layout>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/stream" element={<Stream />} />
              <Route path="/memory" element={<Memory />} />
              <Route path="/query" element={<Query />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/config" element={<Config />} />
              <Route path="*" element={<Overview />} />
            </Routes>
          </Layout>
        </QueryProvider>
      </LiveProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
